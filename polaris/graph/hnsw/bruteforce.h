#pragma once

#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <assert.h>
#include <polaris/core/log.h>

namespace hnswlib {
    template<typename dist_t>
    class BruteforceSearch : public AlgorithmInterface<dist_t> {
    public:
        char *data_;
        size_t maxelements_;
        size_t cur_element_count;
        size_t size_per_element_;

        size_t data_size_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::mutex index_lock;

        std::unordered_map<polaris::vid_t, size_t> dict_external_to_internal;


        BruteforceSearch(SpaceInterface<dist_t> *s)
                : data_(nullptr),
                  maxelements_(0),
                  cur_element_count(0),
                  size_per_element_(0),
                  data_size_(0),
                  dist_func_param_(nullptr) {
        }


        BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
                : data_(nullptr),
                  maxelements_(0),
                  cur_element_count(0),
                  size_per_element_(0),
                  data_size_(0),
                  dist_func_param_(nullptr) {
            loadIndex(location, s);
        }


        BruteforceSearch(SpaceInterface<dist_t> *s, size_t maxElements) {
            maxelements_ = maxElements;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(polaris::vid_t);
            data_ = (char *) malloc(maxElements * size_per_element_);
            if (data_ == nullptr)
                throw std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
            cur_element_count = 0;
        }


        ~BruteforceSearch() {
            free(data_);
        }


        collie::Status addPoint(const void *datapoint, polaris::vid_t label, bool replace_deleted = false) {
            int idx;
            {
                std::unique_lock<std::mutex> lock(index_lock);

                auto search = dict_external_to_internal.find(label);
                if (search != dict_external_to_internal.end()) {
                    idx = search->second;
                } else {
                    if (cur_element_count >= maxelements_) {
                        return collie::Status::resource_exhausted("The number of elements exceeds the specified limit");
                    }
                    idx = cur_element_count;
                    dict_external_to_internal[label] = idx;
                    cur_element_count++;
                }
            }
            memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(polaris::vid_t));
            memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
            return collie::Status::ok_status();
        }

        size_t size() const {
            return cur_element_count;
        }


        collie::Status mark_delete(polaris::vid_t cur_external) {
            size_t cur_c = dict_external_to_internal[cur_external];

            dict_external_to_internal.erase(cur_external);

            polaris::vid_t label = *((polaris::vid_t * )(data_ + size_per_element_ * (cur_element_count - 1) + data_size_));
            dict_external_to_internal[label] = cur_c;
            memcpy(data_ + size_per_element_ * cur_c,
                   data_ + size_per_element_ * (cur_element_count - 1),
                   data_size_ + sizeof(polaris::vid_t));
            cur_element_count--;
            return collie::Status::ok_status();
        }

        collie::Status search(polaris::SearchContext &ctx) const override {
            if (ctx.top_k >= cur_element_count) {
                return collie::Status::invalid_argument("k should be less than the number of elements");
            }
            std::priority_queue<std::pair<dist_t, int>> topResults;
            if (cur_element_count == 0) {
                return collie::Status::ok_status();
            }
            for (int i = 0; i < ctx.top_k; i++) {
                dist_t dist = fstdistfunc_(ctx.query, data_ + size_per_element_ * i, dist_func_param_);
                polaris::vid_t label = *((polaris::vid_t * )(data_ + size_per_element_ * i + data_size_));
                if (!ctx.search_condition->is_in_blacklist(label)) {
                    topResults.push(std::pair<dist_t, int>(dist, i));
                }
            }
            dist_t lastdist = topResults.empty() ? std::numeric_limits<dist_t>::max() : topResults.top().first;
            for (int i = ctx.top_k; i < cur_element_count; i++) {
                dist_t dist = fstdistfunc_(ctx.query, data_ + size_per_element_ * i, dist_func_param_);
                if (dist <= lastdist) {
                    polaris::vid_t label = *((polaris::vid_t * )(data_ + size_per_element_ * i + data_size_));
                    if (!ctx.search_condition->is_in_blacklist(label)) {
                        topResults.push(std::pair<dist_t, int>(dist, i));
                    }
                    if (topResults.size() > ctx.top_k)
                        topResults.pop();

                    if (!topResults.empty()) {
                        lastdist = topResults.top().first;
                    }
                }
            }
            while (!topResults.empty()) {
                auto top = topResults.top();
                polaris::vid_t label = *((polaris::vid_t * )(data_ + size_per_element_ * top.second + data_size_));
                ctx.top_k_queue.emplace_back(label, top.first);
                topResults.pop();
            }

            return collie::Status::ok_status();
        }

        collie::Status get_vector(polaris::vid_t vid, void *vec) const {
            auto search = dict_external_to_internal.find(vid);
            if (search == dict_external_to_internal.end()) {
                return collie::Status::not_found("Vector not found");
            }
            int idx = search->second;
            memcpy(vec, data_ + size_per_element_ * idx, data_size_);
            return collie::Status::ok_status();
        }

        std::priority_queue<std::pair<dist_t, polaris::vid_t >>
        searchKnn(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
            assert(k <= cur_element_count);
            std::priority_queue<std::pair<dist_t, polaris::vid_t >> topResults;
            if (cur_element_count == 0) return topResults;
            for (int i = 0; i < k; i++) {
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                polaris::vid_t label = *((polaris::vid_t * )(data_ + size_per_element_ * i + data_size_));
                if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                    topResults.push(std::pair<dist_t, polaris::vid_t>(dist, label));
                }
            }
            dist_t lastdist = topResults.empty() ? std::numeric_limits<dist_t>::max() : topResults.top().first;
            for (int i = k; i < cur_element_count; i++) {
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                if (dist <= lastdist) {
                    polaris::vid_t label = *((polaris::vid_t * )(data_ + size_per_element_ * i + data_size_));
                    if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                        topResults.push(std::pair<dist_t, polaris::vid_t>(dist, label));
                    }
                    if (topResults.size() > k)
                        topResults.pop();

                    if (!topResults.empty()) {
                        lastdist = topResults.top().first;
                    }
                }
            }
            return topResults;
        }


        collie::Status saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, size_per_element_);
            writeBinaryPOD(output, cur_element_count);

            output.write(data_, maxelements_ * size_per_element_);

            output.close();
            return collie::Status::ok_status();
        }


        collie::Status load(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) override {
            std::ifstream input(location, std::ios::binary);
            std::streampos position;

            readBinaryPOD(input, maxelements_);
            readBinaryPOD(input, size_per_element_);
            readBinaryPOD(input, cur_element_count);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(polaris::vid_t);
            data_ = (char *) malloc(maxelements_ * size_per_element_);
            if (data_ == nullptr) {
                return collie::Status::resource_exhausted("Not enough memory: loadIndex failed to allocate data");
            }

            input.read(data_, maxelements_ * size_per_element_);

            input.close();
            return collie::Status::ok_status();
        }
    };
}  // namespace hnswlib

/*
 *Copyright (c) 2018, Tencent. All rights reserved.
 *
 *Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of elasticfaiss nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 *THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
 *BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 *THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <pthread.h>
#include <stdint.h>

namespace polaris {
    using non_recursive_mutex = std::mutex;
    using LockGuard = std::lock_guard<non_recursive_mutex>;

    enum LockMode {
        INVALID_LOCK, READ_LOCK, WRITE_LOCK
    };

    class RWLock {
    public:
        RWLock() { pthread_rwlock_init(&lock_, nullptr); }

        bool lock(LockMode mode) {
            switch (mode) {
                case READ_LOCK: {
                    return 0 == pthread_rwlock_rdlock(&lock_);
                }
                case WRITE_LOCK: {
                    return 0 == pthread_rwlock_wrlock(&lock_);
                }
                default: {
                    return false;
                }
            }
        }

        bool unlock(LockMode mode) { return 0 == pthread_rwlock_unlock(&lock_); }

        ~RWLock() { pthread_rwlock_destroy(&lock_); }

    private:
        pthread_rwlock_t lock_;
    };

    class ReadLock {
    public:
        explicit ReadLock(RWLock &lock) : lock_(lock) {}

        void lock() { lock_.lock(READ_LOCK); }

        void unlock() { lock_.unlock(READ_LOCK); }

    private:
        RWLock &lock_;
    };

    class WriteLock {
    public:
        explicit WriteLock(RWLock &lock) : lock_(lock) {}

        void lock() { lock_.lock(WRITE_LOCK); }

        void unlock() { lock_.unlock(WRITE_LOCK); }

    private:
        RWLock &lock_;
    };

}  // namespace polaris

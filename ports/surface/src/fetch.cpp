#include "fetch.h"

#include "json.h"

#include <curl/curl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace hx {

namespace {

constexpr int ATTEMPTS = 6;

struct Job {
    size_t cell = 0;
    std::string url, body;
    int attempt = 0;
    std::chrono::steady_clock::time_point ready{};
};

size_t collect(char *data, size_t size, size_t n, void *user) {
    static_cast<std::string *>(user)->append(data, size * n);
    return size * n;
}

std::string strip_slashes(const std::string &s) {
    const size_t a = s.find_first_not_of('/');
    if (a == std::string::npos)
        return "";
    return s.substr(a, s.find_last_not_of('/') - a + 1);
}

std::string source_url(const std::string &bucket, const std::string &source) {
    const size_t sep = source.find("//");
    if (sep == std::string::npos)
        throw std::runtime_error("store source is not PATH//LEVEL: " + source);
    std::string base = bucket;
    while (!base.empty() && base.back() == '/')
        base.pop_back();
    return base + "/" + strip_slashes(source.substr(0, sep)) + "/" + source.substr(sep + 2);
}

void make_dirs(const std::string &path) {
    for (size_t at = path.find('/', 1); at != std::string::npos; at = path.find('/', at + 1))
        mkdir(path.substr(0, at).c_str(), 0755);
    mkdir(path.c_str(), 0755);
}

void append_all(int fd, const std::string &body) {
    size_t done = 0;
    while (done < body.size()) {
        const ssize_t n = ::write(fd, body.data() + done, body.size() - done);
        if (n <= 0)
            throw std::runtime_error("cannot append to bricks.bin");
        done += (size_t)n;
    }
}

}

int64_t fetch_missing(
    const std::string &folder, const Box &box, const std::string &bucket, int connections) {
    BrickStore store;
    store.open(folder);
    if (store.source.empty())
        return 0;
    const std::string base = source_url(bucket, store.source);
    std::vector<Job> jobs;
    for (int64_t i = box[0][0] / store.brick[0];
        i < (box[0][1] + store.brick[0] - 1) / store.brick[0]; ++i)
        for (int64_t j = box[1][0] / store.brick[1];
            j < (box[1][1] + store.brick[1] - 1) / store.brick[1]; ++j)
            for (int64_t k = box[2][0] / store.brick[2];
                k < (box[2][1] + store.brick[2] - 1) / store.brick[2]; ++k) {
                if (i >= store.grid[0] || j >= store.grid[1] || k >= store.grid[2])
                    continue;
                const size_t cell = (size_t)((i * store.grid[1] + j) * store.grid[2] + k);
                if (store.offsets[cell] != -1)
                    continue;
                Job job;
                job.cell = cell;
                job.url = base + "/" + std::to_string(i) + "/" + std::to_string(j) + "/" +
                    std::to_string(k);
                jobs.push_back(std::move(job));
            }
    if (jobs.empty())
        return 0;
    const size_t brick_bytes = (size_t)(store.brick[0] * store.brick[1] * store.brick[2]);
    const std::string bricks = folder + "/bricks.bin";
    const int fd = ::open(bricks.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0)
        throw std::runtime_error("cannot open " + bricks);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURLM *multi = curl_multi_init();
    std::vector<CURL *> handles, idle;
    for (int c = 0; c < std::max(1, connections); ++c)
        handles.push_back(curl_easy_init());
    idle = handles;
    std::deque<size_t> waiting;
    for (size_t a = 0; a < jobs.size(); ++a)
        waiting.push_back(a);
    std::unordered_map<CURL *, size_t> active;
    int64_t got = 0, failed = 0;
    std::string failure;
    while (!waiting.empty() || !active.empty()) {
        const auto now = std::chrono::steady_clock::now();
        for (size_t scan = waiting.size(); scan > 0 && !idle.empty(); --scan) {
            const size_t at = waiting.front();
            waiting.pop_front();
            if (jobs[at].ready > now) {
                waiting.push_back(at);
                continue;
            }
            CURL *easy = idle.back();
            idle.pop_back();
            active[easy] = at;
            jobs[at].body.clear();
            curl_easy_setopt(easy, CURLOPT_URL, jobs[at].url.c_str());
            curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, collect);
            curl_easy_setopt(easy, CURLOPT_WRITEDATA, &jobs[at].body);
            curl_easy_setopt(easy, CURLOPT_TIMEOUT, 60L);
            curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
            curl_multi_add_handle(multi, easy);
        }
        int running = 0;
        curl_multi_perform(multi, &running);
        curl_multi_poll(multi, nullptr, 0, 100, nullptr);
        int left = 0;
        while (CURLMsg *msg = curl_multi_info_read(multi, &left)) {
            if (msg->msg != CURLMSG_DONE)
                continue;
            CURL *easy = msg->easy_handle;
            const CURLcode result = msg->data.result;
            long status = 0;
            curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status);
            curl_multi_remove_handle(multi, easy);
            idle.push_back(easy);
            const size_t at = active[easy];
            active.erase(easy);
            Job &job = jobs[at];
            if (result == CURLE_OK && (status == 200 || status == 206)) {
                if (job.body.size() != brick_bytes) {
                    ++failed;
                    failure = job.url + ": " + std::to_string(job.body.size()) +
                        " bytes, not a raw brick";
                } else {
                    const off_t offset = lseek(fd, 0, SEEK_END);
                    append_all(fd, job.body);
                    store.offsets[job.cell] = (int64_t)offset;
                    ++got;
                }
                job.body = std::string();
            } else if (result == CURLE_OK && (status == 403 || status == 404)) {
                store.offsets[job.cell] = -2;
            } else if (++job.attempt >= ATTEMPTS) {
                ++failed;
                failure = job.url + ": " +
                    (result != CURLE_OK ? std::string(curl_easy_strerror(result))
                                        : "HTTP " + std::to_string(status));
            } else {
                job.ready = std::chrono::steady_clock::now() +
                    std::chrono::seconds(std::min(30, 1 << (job.attempt - 1)));
                waiting.push_back(at);
            }
        }
    }
    for (CURL *easy : handles)
        curl_easy_cleanup(easy);
    curl_multi_cleanup(multi);
    curl_global_cleanup();
    ::close(fd);
    const std::string table = folder + "/offsets.npy", temporary = folder + "/offsets.tmp.npy";
    save_npy(temporary, "<i8", {store.grid[0], store.grid[1], store.grid[2]}, store.offsets.data(),
        store.offsets.size() * sizeof(int64_t));
    if (std::rename(temporary.c_str(), table.c_str()) != 0)
        throw std::runtime_error("cannot replace " + table);
    if (failed)
        throw std::runtime_error(
            std::to_string(failed) + " bricks of the box could not be fetched, last " + failure);
    return got;
}

bool partial_store(const std::string &folder) { return exists(folder + "/planned.json"); }

void create_store(const std::string &folder, const std::string &source, const std::string &bucket) {
    const std::string url = source_url(bucket, source) + "/.zarray";
    std::string body;
    long status = 0;
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL *easy = curl_easy_init();
    curl_easy_setopt(easy, CURLOPT_URL, url.c_str());
    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, collect);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(easy, CURLOPT_TIMEOUT, 60L);
    const CURLcode result = curl_easy_perform(easy);
    curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(easy);
    curl_global_cleanup();
    if (result != CURLE_OK || status != 200)
        throw std::runtime_error("cannot read " + url);
    const Json z = Json::parse(body);
    const bool raw = z["dtype"].str() == "|u1" && z["compressor"].kind == Json::Null &&
        (!z.has("filters") || z["filters"].kind == Json::Null) &&
        (!z.has("order") || z["order"].str() == "C") && z.has("dimension_separator") &&
        z["dimension_separator"].str() == "/" && z["shape"].size() == 3;
    if (!raw)
        throw std::runtime_error(source + " is not a raw uint8 zarr with / separated chunks");
    std::array<int64_t, 3> shape{}, brick{}, grid{};
    for (int d = 0; d < 3; ++d) {
        shape[d] = (int64_t)z["shape"][d].num();
        brick[d] = (int64_t)z["chunks"][d].num();
        grid[d] = (shape[d] + brick[d] - 1) / brick[d];
    }
    make_dirs(folder);
    write_text(folder + "/meta.json",
        "{\"shape\": [" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " +
            std::to_string(shape[2]) + "], \"brick\": [" + std::to_string(brick[0]) + ", " +
            std::to_string(brick[1]) + ", " + std::to_string(brick[2]) +
            "], \"source\": " + quote(source) + "}");
    const std::vector<int64_t> unknown((size_t)(grid[0] * grid[1] * grid[2]), -1);
    save_npy(folder + "/offsets.npy", "<i8", {grid[0], grid[1], grid[2]}, unknown.data(),
        unknown.size() * sizeof(int64_t));
    write_text(folder + "/bricks.bin", "");
}

}

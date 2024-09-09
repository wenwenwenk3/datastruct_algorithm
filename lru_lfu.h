//
// Created by kai.chen on 2021/12/12.
//
//      LRU & LFU
//

#ifndef DATASTRUCT_ALGORITHM_LRU_LFU_H
#define DATASTRUCT_ALGORITHM_LRU_LFU_H

template typename <K, V>
struct DLinkedNode {
    K key;
    V value;
    DLinkedNode* prev;
    DLinkedNode* next;
    DLinkedNode(): key(0), value(0), prev(nullptr), next(nullptr) {}
    DLinkedNode(K _key, V _value): key(_key), value(_value), prev(nullptr), next(nullptr) {}
};


// 1. LRU
// 思路: 为了实现O(1)复杂度的读取效率需要用哈希
//      又为了实现过期淘汰策略需要用链表
//  put流程：
//      (1).若key已存在： 修改val值，将key提升为最近使用
//      (2).若key不存在： 创建新节点并插入哈希表和链表，判断size大小超过cap
//              若size超过cap，过期删除链表尾部元素
//  get流程：
//      (1).通过哈希表判断 若key存在，提升为最近使用，返回
class LRUCache {
private:
    unordered_map<int, DLinkedNode*> cache;
    DLinkedNode* head;
    DLinkedNode* tail;
    int size;
    int capacity;

public:
    LRUCache(int _capacity): capacity(_capacity), size(0) {
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head->next = tail;
        tail->prev = head;
    }

    int get(int key) {
        if (!cache.count(key)) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        DLinkedNode* node = cache[key];
        moveToHead(node);
        return node->value;
    }

    void put(int key, int value) {
        if (!cache.count(key)) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode* node = new DLinkedNode(key, value);
            // 添加进哈希表
            cache[key] = node;
            // 添加至双向链表的头部
            addToHead(node);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode* removed = removeTail();
                // 删除哈希表中对应的项
                cache.erase(removed->key);
                // 防止内存泄漏
                delete removed;
                --size;
            }
        }
        else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            DLinkedNode* node = cache[key];
            node->value = value;
            moveToHead(node);
        }
    }

    void addToHead(DLinkedNode* node) {
        node->prev = head;
        node->next = head->next;
        head->next->prev = node;
        head->next = node;
    }

    void removeNode(DLinkedNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void moveToHead(DLinkedNode* node) {
        removeNode(node);
        addToHead(node);
    }

    DLinkedNode* removeTail() {
        DLinkedNode* node = tail->prev;
        removeNode(node);
        return node;
    }
};



// 2. LFU
// https://github.com/shichangzhi/fucking-algorithm-book/blob/main/%E7%AC%AC3%E7%AB%A0-%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E7%B3%BB%E5%88%97/3.2-%E5%B1%82%E5%B1%82%E6%8B%86%E8%A7%A3%EF%BC%8C%E5%B8%A6%E4%BD%A0%E6%89%8B%E5%86%99LFU%E7%AE%97%E6%B3%95.md
class LFUCache {
private:
    // key 到 val 的映射，我们后文称为 KV 表
    HashMap<Integer, Integer> keyToVal;
    // key 到 freq 的映射，我们后文称为 KF 表
    HashMap<Integer, Integer> keyToFreq;
    // freq 到 key 列表的映射，我们后文称为 FK 表
    HashMap<Integer, LinkedHashSet<Integer>> freqToKeys;

    // 记录最小的频次
    int minFreq;
    // 记录 LFU 缓存的最大容量
    int cap;
public:
    LFUCache(int capacity) {
        keyToVal = new HashMap<>();
        keyToFreq = new HashMap<>();
        freqToKeys = new HashMap<>();
        this.cap = capacity;
        this.minFreq = 0;
    }
    int get(int key) {}

    void put(int key, int val) {}
}


// 3. HashMap
// https://blog.csdn.net/Mr_Garfield__/article/details/79418729?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165090311416781483733973%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=165090311416781483733973&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-79418729.nonecase&utm_term=%E5%93%88%E5%B8%8C&spm=1018.2226.3001.4450
// 哈希函数设计原则：
//    - 哈希函数的定义域必须包括需要存储的全部关键码，而如果散列表允许有m个地址时，其值域必须在0到m-1之间
//    - 哈希函数计算出来的地址能均匀分布在整个空间中
//    - 哈希函数应该比较简单
class MyHashMap1 { // 开链法
private:
    vector<list<pair<int, int>>> data;
    static const int base = 769;
    static int hash(int key) {
        return key % base;
    }
public:
    /** Initialize your data structure here. */
    MyHashMap1(): data(base) {}

    /** value will always be non-negative. */
    void put(int key, int value) {
        int h = hash(key);
        for (auto it = data[h].begin(); it != data[h].end(); it++) {
            if ((*it).first == key) {
                (*it).second = value;
                return;
            }
        }
        data[h].push_back(make_pair(key, value));
    }

    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    int get(int key) {
        int h = hash(key);
        for (auto it = data[h].begin(); it != data[h].end(); it++) {
            if ((*it).first == key) {
                return (*it).second;
            }
        }
        return -1;
    }

    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    void remove(int key) {
        int h = hash(key);
        for (auto it = data[h].begin(); it != data[h].end(); it++) {
            if ((*it).first == key) {
                data[h].erase(it);
                return;
            }
        }
    }
};

size_t GetNextPrimeSize(size_t num){
    // 使用素数表对齐做哈希表的容量，降低哈希冲突
    const int PrimeSize = 28;
    static const unsigned long PrimeList[] ={
                    53ul, 97ul, 193ul, 389ul, 769ul,
                    1543ul, 3079ul, 6151ul, 12289ul, 24593ul,
                    49157ul, 98317ul, 196613ul, 393241ul, 786433ul,
                    1572869ul, 3145739ul, 6291469ul, 12582917ul, 25165843ul,
                    50331653ul, 100663319ul, 201326611ul, 402653189ul, 805306457ul,
                    1610612741ul, 3221225473ul, 4294967291ul
            };
    for (size_t i = 0; i < PrimeSize; i++){
        if (num * 10 / PrimeList[i] <= 7)
            return PrimeList[i];
    }
    return PrimeList[PrimeSize - 1];        //最大返回容量约4G
}


class MyHashMap2 { // 开放地址法，线性探测
public:
    MyHashMap() {
        hashTable = vector<pair<int, int>>(N, {-1, -1});
    }

    int find(int key) {
        int k = key % N;
        while (hashTable[k].first != key && hashTable[k].first != -1) {
            k = (k + 1) % N;
        }

        return k;
    }

    void put(int key, int value) {
        auto k = find(key);
        hashTable[k] = {key, value};
    }

    int get(int key) {
        auto k = find(key);
        if (hashTable[k].first == -1) {
            return -1;
        }

        return hashTable[k].second;
    }

    void remove(int key) {
        auto k = find(key);
        if (hashTable[k].first != -1) {
            hashTable[k].first = -2; // Mark as deleted (use a different value with -1)
        }
    }

private:
    const static int N = 20011;
    vector<pair<int, int>> hashTable;
}; // 时间复杂度：一般情况下复杂度为 O(1)，极端情况下为 O(n)
// 空间复杂度：O(1)



#endif //DATASTRUCT_ALGORITHM_LRU_LFU_H
//
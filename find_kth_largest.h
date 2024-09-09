//
// Created by kai.chen on 2021/12/11.
//  1. 找数组的第K大元素
//      变体1.1 数组中出现次数超过一半的数，其实就是求第n/2大的数
//      变体1.2 前 k 个高频元素
//
//  2. 找数据流的中位数
//      2.1 数据流的第k大的数
//      2.2 滑动窗口的中位数
//  3. 找两个升序数组的中位数

#ifndef DATASTRUCT_ALGORITHM_FIND_KTH_LARGEST_H
#define DATASTRUCT_ALGORITHM_FIND_KTH_LARGEST_H
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;

// 1. 数组的第K大元素
// 三问：
// (1). 数组的元素个数N够大吗？
//          N不大直接排序（基本排序n^2，高级排序nlogn）或 直接部分选择排序 n*k，堆调整 nlogK （N超级大时而K较小适合）
// (2). K够大吗？
//          k比较小时候可以 基于选择排序的部分排 O(n*k)，
//          k够大的话借助 "快排分割" 平均时间复杂度: O(N)，最差n^2. 因为每次划分可能都是当前最值，而划分操作需要O(n)时间复杂度
// (3). 元素分布够集中吗？
//          借助计数排序 最差O(N)
// (4). 由于顺序统计量这个问题太经典，算法导论有介绍一种最差为O(N)的 SELECT 算法
//          还是基于快排分割的思想，但优化了主元的选择。高区和低区的判断优化了最差情况。
//    理论上存在 线性算法，但常数项特别大，实际应用效果并不好
void ajust_down(vector<int>& nums, int n, int root){
    int parent = root;
    int child = 2*parent+1;
    while(child < n){
        if(child+1 < n && nums[child+1] > nums[child]){ // 选出左右子节点较大值
            child++;
        }
        if(nums[child] > nums[parent]){ // 建大根堆，大的元素需要往上调整
            swap(nums[child], nums[parent]);
            parent = child;
            child = 2*parent+1;
        }else{
            break;
        }
    }
}
int findKthLargest_V1(vector<int>& nums, int k){ // 基于堆调整 n logk
    if(k > nums.size() || k < 1) return -1;
    // 建大堆，
    // make_heap(nums.begin(), nums.end());
    int sz = nums.size();
    for(int i = (sz-1-1)/2; i >= 0; i--){ // n/2 * logN
        ajust_down(nums, sz, i);
    }

    // 依次把最大的元素交换到最后，k次之后就是第k大
    for(int i = sz-1; i >= 0; i--){
        if(--k == 0){
            return nums[0];
        }
        swap(nums[0], nums[i]);
        ajust_down(nums, i, 0);
    }
    return -1;
}

int findKthLargest_V2(vector<int>& nums, int k){ // 基于选择排序的部分排 O(n*k)
    if(k > nums.size() || k < 1) return -1;
    for(int i = 0; i < k; i++){
        int minIdx = i;
        for(int j = i+1; j < nums.size(); j++){
            if(nums[j] > nums[minIdx]){
                minIdx = j;
            }
        }
        swap(nums[i], nums[minIdx]);
    }
    return nums[k-1];
}

int midOfRange(vector<int>& nums, int left, int right){
    int mid = left + ((right-left)>>1);
    if(nums[left] < nums[right]){
        if(nums[mid] < nums[left]) return left;
        else if(nums[mid] < nums[right]) return mid;
        else return right;
    }
    else {
        if(nums[mid] < nums[right]) return right;
        else if (nums[mid] > nums[left]) return mid;
        else return left;
    }
}

int partitionv(vector<int>& nums, int left, int right){
    int index = midOfRange(nums, left, right);
    swap(nums[index], nums[right]);
    int l = left, r = right;
    int key = nums[right];
    while(l < r){
        while(l<r & nums[l] <= key) l++;
        while(l<r & nums[r] >= key) r--;
        swap(nums[l], nums[r]);
    }
    swap(nums[l], nums[right]);
    return l;
}

int quick_select(vector<int>& nums, int l, int r, int idx){
    int q = partitionv(nums, l, r);
    if(q == idx) {
        return nums[q];
    }
    else if(q < idx){
        return quick_select(nums, q+1, r, idx);
    }
    // else if q > idx
    return quick_select(nums, l, q-1, idx);
}

int findKthLargest_V3(vector<int>& nums, int k) { // 基于快排的分割
    if(k > nums.size() || k < 1) return -1;
    return quick_select(nums, 0, nums.size()-1, nums.size()-k);
}


const int MAX = 1000;
int findKthLargest_V4(vector<int>& nums, int k) { // 基于计数排序的方法
    int count[MAX] = {0};
    for(int i = 0; i < nums.size()-1; i++){
        count[nums[i]]++;
    }

    int sumCount = 0;
    int i = MAX-1;
    for(; i >= 0; --i){ // 从后往前找，找倒数第k个元素
        sumCount += count[i];
        if(sumCount >= k) break;
    }
    return i;
}


void testKlargest(){
    // int a[] = {3,2,5,8,4,7,6,9,1};
    int a[] = {-1,2,0};
    vector<int> nums(a, a+sizeof(a)/sizeof(a[0]));
    int res = findKthLargest_V1(nums, 3);
    // int res = findKthLargest_V2(nums, 2);
    // int res = findKthLargest_V3(nums, 4);
    // int res = findKthLargest_V4(nums, 2);
    cout<<"klargest result is "<<res<<endl;
}

// 变体1.1 数组中出现次数超过一半的数
// 思路
//     (1)，转化一下其实就是求第n/2大的数, 快排分割法能做到O(N)
//     (2), 保存一个数字，每次数字相同就+1， 不同就-1，次数为0就更新数字
bool checkMoreThanHalfTimes(vector<int>& nums, int value){
    int t = 0;
    for(int item : nums){
        t += item == value;
    }
    return t*2 > nums.size();
}
//方法(1)
int moreThanHalfNum1(vector<int>& nums){
    int n = nums.size();
    int kth = quick_select(nums, 0, n-1, n/2);
    if (!checkMoreThanHalfTimes(nums, kth)){
        return 0;
    }
    return kth;
}
//方法(2)
int moreThanHalfNum2(vector<int>& nums){
    int x = nums[0]; // 保存一个数字
    int t = 1; // 数字出现次数
    for(int i = 1; i <nums.size(); i++){ // 保证算出x是出现次数最多的数
        int currItem = nums[i];
        if(t == 0) { // 换数字
            x = currItem;
            t = 1;
        }
        else if(currItem == x){
            t++;
        }else if(currItem != x){
            t--;
        }
    }
    if (!checkMoreThanHalfTimes(nums, x)){
        return 0;
    }
    return x;
}

// 1.2 Top K 找最小的k个数
vector<int> getLeastNumbers(vector<int>& arr, int k) {
    if (arr.size() == 0 || k == 0) return {}; // 排除 0 的情况
    vector<int> vec(k, 0);
    priority_queue<int> q; // c++的q是大根堆
    for (int i = 0; i < (int)arr.size(); ++i) {
        if(i < k){
            q.push(arr[i]);
        }
        else if (arr[i] < q.top()) {
            q.pop();
            q.push(arr[i]);
        }
    }
    for (int i = 0; i < k; ++i) {
        vec[i] = q.top();
        q.pop();
    }
    return vec;
} // 时间：O(Nlogk), 每次堆调整的时间复杂度为logk
// 空间：大根堆里最多 k 个数。

// 变体1.2 前 k 个高频元素
// 思路：
//     (1)用哈希表记录每个数字出现的次数，并保存 pair<数字, 一个出现次数> 数组。
//        直接的思路是对这个数组按出现次数排序，找到第k个就是结果。但时间复杂度O( N*logN)
//      可以 维护一个最小堆 来优化时间复杂度到 O(N * logK)
//     (2)也可以转化为 数组的第k大元素用快排分割做，最差O(N^2)
//     (3)哈希表统计次数后进行 桶排序
struct cmp1{
    bool operator()(pair<int, int>& m, pair<int, int>& n){
        return m.second > n.second;
    }
};
bool cmp2(pair<int, int>& m, pair<int, int>& n) {
    return m.second > n.second;
}
// 堆排法 NlogK
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (auto& v : nums) {
        freq[v]++;
    }
    // pair 的第一个元素代表数组的值，第二个元素代表了该值出现的次数
    // 利用priority_queue可以内部实现堆调整
    // priority_queue<Type, Container, Functional>
    priority_queue< pair<int, int>, vector<pair<int, int>>, cmp1>  q;
    //或用 priority_queue< pair<int, int>, vector<pair<int, int>>, decltype(&cmp2)>  q(cmp2);
    for (auto& item : freq) {
        if (q.size() == k) {
            if (q.top().second < item.second) {
                q.pop();
                q.emplace(item);
            }
        } else {
            q.emplace(item);
        }
    }
    vector<int> res;
    while (!q.empty()) {
        res.emplace_back(q.top().first);
        q.pop();
    }
    return res;
}
// 前k个高频单词 朴素排序法NlogN
vector<string> topKFrequent(vector<string>& words, int k) {
    unordered_map<string, int> freq;
    for(const auto& word: words){
        ++freq[word];
    }
    vector<pair<string, int>> freqArr;
    for(const auto& item: freq){
        freqArr.push_back(make_pair(item.first, item.second));
    }

    sort(freqArr.begin(), freqArr.end(), [](pair<string, int>& a, pair<string, int>& b){
        return a.second==b.second? a.first<b.first : a.second>b.second;
    });

    vector<string> res(k);
    for(int i = 0; i < k; i++){
        res[i] = freqArr[i].first;
    }
    return res;
}

// 前 K 个高频元素 桶排序法 time O(N) space O(N)
/*
func topKFrequent(nums []int, k int) []int {
	if len(nums) == 0 {
		return []int{}
	}
	ht := make(map[int]int, 0)
	for _, v := range nums {
		ht[v]++
	}
	// 使用「桶排序」来进行频次筛选
	buckets := make([][]int, len(nums)+1)
	for num, cnt := range ht {
		if len(buckets[cnt]) == 0 {
			buckets[cnt] = make([]int, 0)
		}
		buckets[cnt] = append(buckets[cnt], num)
	}

	ans := make([]int, 0)
	for i := len(buckets) - 1; i >= 0; i-- {
		// 空桶，跳过
		if len(buckets[i]) == 0 {
			continue
		}
		ans = append(ans, buckets[i]...)
		// 已经获得 top k，则停止筛选。
		if len(ans) == k {
			break
		}
	}

	return ans
}*/


//  2. 找数据流的中位数
// 两个堆
class MedianFinder {
public:
    priority_queue<int, vector<int>, less<int>> queMin;
    priority_queue<int, vector<int>, greater<int>> queMax;

    void addNum(int num) {
        if (queMin.empty() || num <= queMin.top()) {
            queMin.push(num);
            // 保持小根堆的元素数量不超过 大根堆元素数量+1
            if (queMax.size() + 1 < queMin.size()) {
                queMax.push(queMin.top());
                queMin.pop();
            }
        } else {
            queMax.push(num);
            // 同理 保持大根堆的元素数量不超过 小根堆元素数量
            if (queMax.size() > queMin.size()) {
                queMin.push(queMax.top());
                queMax.pop();
            }
        }
    }

    double findMedian() {
        if (queMin.size() > queMax.size()) {
            return queMin.top();
        }
        return (queMin.top() + queMax.top()) / 2.0;
    }
};
// 数组实现堆
class MedianFinder2 {
public:
    vector<int> minHeap;
    vector<int> maxHeap;

    void addNum(int num) {
        if (minHeap.empty() || num > minHeap[0]) {
            minHeap.push_back(num);
            if (maxHeap.size() + 1 < minHeap.size()) {
                maxHeap.push_back(minHeap[0]);
                push_heap(maxHeap.begin(), maxHeap.end(), greater<int>());

                pop_heap(minHeap.begin(), minHeap.end(), less<int>());
                minHeap.pop_back();
            }
        } else {
            maxHeap.push_back(num);
            if (maxHeap.size() > minHeap.size()) {
                minHeap.push_back(maxHeap[0]);
                push_heap(minHeap.begin(), minHeap.end(),less<int>());

                pop_heap(maxHeap.begin(), maxHeap.end(), greater<int>());
                maxHeap.pop_back();
            }
        }
    }

    double findMedian() {
        if (minHeap.size() > maxHeap.size()) {
            return minHeap[0];
        }
        return (minHeap[0] + maxHeap[0]) / 2.0;
    }
};
//进阶 1
// 如果数据流中所有整数都在 0 到 100 范围内，那么我们可以利用计数排序统计每一类数的数量，并使用双指针维护中位数。
//进阶 2
// 如果数据流中 99% 的整数都在 0 到 100 范围内，那么我们依然利用计数排序统计每一类数的数量，并使用双指针维护中位数。
// 对于超出范围的数，我们可以单独进行处理，建立两个数组，分别记录小于 0 的部分的数的数量和大于100 的部分的数的数量即可。当小部分时间，中位数不落在区间 [0,100] 中时，我们在对应的数组中暴力检查即可。

void test_MedianFinder(){
    MedianFinder mf;
    for(int i=1; i <= 100; i++){
        mf.addNum(i);
    }
    cout<<mf.findMedian()<<endl;
}

// 2.1 数据流的第k大的数
class KthLargest {
public:
    priority_queue<int, vector<int>, greater<int>> q;
    int k;
    KthLargest(int k, vector<int>& nums) {
        this->k = k;
        for (auto& x: nums) {
            add(x);
        }
    } // 初始化时间复杂度 O(n*logk)

    int add(int val) {
        q.push(val);
        if (q.size() > k) {
            q.pop();
        }
        return q.top();
    } //
};

// 2.2 滑动窗口的中位数
// 这题hard不要求掌握: https://leetcode.cn/problems/sliding-window-median/solution/hua-dong-chuang-kou-zhong-wei-shu-by-lee-7ai6/


// 3. 找两个升序数组的中位数
//  思路：先合并为一个升序数组，找中位值。其实就是合并两个升序数组
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int len1 = nums1.size();
    int len2 = nums2.size();
    vector<int> res(len1+len2,0);
    int idx = 0;
    int i = 0,j=0;
    while(i < len1 &&j<len2){
        if(nums1[i]<nums2[j]){
            res[idx] = nums1[i];
            idx++;
            i++;
        }
        else if (nums1[i]>nums2[j]){
            res[idx] = nums2[j];
            idx++;
            j++;
        }
        else if (nums1[i]==nums2[j]){
            res[idx] = nums1[i];
            res[idx+1] = nums2[j];
            idx = idx+2;
            i++;
            j++;
        }
    }
    if(i < len1){
        for(int t = i;t<len1;t++){
            res[idx++] = nums1[t];
        }
    }
    if(j < len2){
        for(int t = j;t<len2;t++){
            res[idx++] = nums2[t];
        }
    }
    return (len1+len2)%2 == 0 ? double(res[(len1+len2)/2]+res[(len1+len2)/2-1])/2 : double(res[(len1+len2)/2]);
}


#endif //DATASTRUCT_ALGORITHM_FIND_KTH_LARGEST_H

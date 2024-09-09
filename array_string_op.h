//
// Created by kai.chen on 2021/12/19.
//
//      1. 0～n-1 中缺失的数字
//      2. 有序数组的缺失元素
//      3. 和为k的子数组
//          // 3.1 和为k的最短非空子数组
//          // 3.2 子数组最小值之和
//
//      4. 删掉一个元素以后全为1的最长子数组 见sliding_window.h
//      5. 统计「优美数组」 见sliding_window.h
//
//      6. 第一个只出现一次的字符
//      7. 数组中的逆序对
//      8. 绝对差值和
//
//       - 字符串转整数、计算器  见calculater.h
//       - 字符串解码  见calculater.h
//       - 版本号判断   见calculater.h
//      - 字符解码方法,  见dp_1d.h
//      - 数字的可能的翻译方法总数  见dp_1d.h
//
//      // 交错字符串 见 dp_hd.h 5.
#ifndef DATASTRUCT_ALGORITHM_ARRAY_STRING_OP_H
#define DATASTRUCT_ALGORITHM_ARRAY_STRING_OP_H

// 1. 0～n-1 中缺失的数字
// 思路： 先排序，二分法：
//  计算中点 mid = left + (right-left)/2 ，其中mid 为向下取整除法；
//  若 nums[mid] == mid ，则 “右子数组的首位元素” 一定在闭区间 [mid + 1, right]中，因此执行 left = m + 1；
//  若 nums[mid] != mid ，则 “左子数组的末位元素” 一定在闭区间 [left, m - 1] 中，因此执行 right = m - 1；
int missingNumber(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    int left = 0, right = nums.size() - 1;
    while(left <= right) {
        int mid = left + (right-left)/2;
        if(nums[mid] == mid) {
            left = mid + 1;
        }else{
            right = mid - 1;
        }
    }
    if(left>=nums.size() || nums[left]==left){ // 没有缺失数字
        return nums.size();
    }
    return left;
} // O(nlogn)

// 思路位运算：再添加n-1个数进行异或，最后剩那个数就是结果
int missingNumber2(vector<int>& nums) {
    int res = 0;
    int n = nums.size() + 1;
    for (int i = 0; i < n - 1; i++) {
        res ^= nums[i];
    }
    for (int i = 0; i <= n - 1; i++) {
        res ^= i;
    }
    return res;
}


/* 数学 前n项和，
 * func missingNumber(nums []int) int {
    n := len(nums)+1
    sum := 0
    for _, x := range nums {
        sum+=x
    }
    return (n-1)*n/2 - sum
}*/

// 2. 有序数组的缺失元素
// 题目描述：给出一个有序数组 A，数组中的每个数字都是 独一无二的，找出从数组最左边开始的第 K 个缺失数字。
//      如输入：A = [4,7,9,10], K = 1 输出：5；  输入：A = [4,7,9,10], K = 3 输出：8
// 思路1：明显可以通过直接遍历做到
// 思路2：需要利用到有序的性质，可以二分遍历
//思路：
//1 这题明显是可以遍历做完的。但是对于有序数组，遍历O(N)解法往往不够，优先考虑二分查找。
//2 注意一点，从数组最左边开始到索引i之间缺少的数字数目可以用：nums[i] - nums[0] - i表示。
//3 那么，可以在二分时判断第mid索引缺少与k比较。找到最左边缺少数目=k的索引。代码如下：
int missingElementK(vector<int>& nums, int k) {
        int left = 0;
        int right = nums.size()-1;

        // 最后一个元素到第一个元素之间缺少的数字数目
        int idx = nums[right] - nums[0] - right;
        if(k > idx){
            return nums[right]+(k-idx);
        }
        while(left <= right){
            int mid = left+(right-left)/2;
            idx = nums[mid] - nums[0] - mid; //到索引i之间缺少的数字数目
            if(idx > k){
                right = mid-1;
            }else if(idx < k){
                left = mid+1;
            }else{  // 相当于是等于的时候，需要找到最左边的节点值;
                right = mid-1;
            }
        }
        return nums[left] + k-(nums[left] - nums[0] - (left));
}


// 3. 和为k的子数组个数
// 题目描述：整数数组 nums 和一个整数 k ，请你统计并返回该数组中和为 k 的连续子数组的个数。
// 思路： 暴力法：考虑以 i 结尾和为 k 的连续子数组个数，只需要找符合条件的下标j的个数
int subarraySum(vector<int>& nums, int k) {
    int count = 0;
    for (int i = 0; i < nums.size(); ++i) {
        int sum = 0;
        for (int j = i; j >= 0; --j) {
            sum += nums[j]; // 每次直接加和
            if (sum == k) { // 判断区间[j,i]和是否和为k
                count++;
            }
        }
    }
    return count;
} // 时间复杂度O(n^2), 空间O(1)

// 更一般的思路，前缀和
//      这里的技巧在于，当我们知道 [j,i]子数组的和，就能 O(1)时间推出[j-1,i]的和
int subarraySum2(vector<int>& nums, int k) {
    int n = nums.size();
    int* presum = new int[n+1];
    presum[0] = 0;
    for(int i = 1; i < n+1; i++){ // 构造前缀和
        presum[i] = presum[i-1] + nums[i-1];
    }

    int count = 0;
    // 穷举所有子数组 [j ..i-1]
    for(int i = 1; i <= n; i++){
        for(int j= 0; j <i; j++){
            // sum of nums[j .. i-1]
            if(presum[i] - presum[j] == k){
                count++;
            }
        }
    }
    return count;
}  // 时间复杂度O(n^2), 空间O(N)

// 前缀和 + 哈希表优化
// 前面的思路都是 对所有以i结尾，顺序找符合条件的下标j。
//    这里找下标j其实是可以用哈希表优化的:
//     因为 pre[i]=pre[i−1]+nums[i]，那么 [j..i]数组的和为k相当于 pre[i]−pre[j−1]==k
//     即 符合条件的下标j 必有： pre[j−1]==pre[i]−k
// 检查在当前数之前，有多少个前缀和等于 preSum - k 的呢
int subarraySum3(vector<int>& nums, int k) {
    int n = nums.size();
    int* presum = new int[n+1];
    presum[0] = 0;
    for(int i = 1; i < n+1; i++){ // 构造前缀和
        presum[i] = presum[i-1] + nums[i-1];
    }

    int count = 0;
    // key：前缀和，value：key 所对应的前缀和的个数
    unordered_map<int, int> premap;
    premap[0] = 1;
    for (int i = 1; i < n+1; i++) {
        count += premap[presum[i] - k];
        premap[presum[i]]++;
    }

    return count;
} // 时间复杂度O(n), 空间O(N)

// 和为k的最短非空子数组


// 4. 删掉一个元素以后全为 1 的最长子数组
// 输入：nums = [1,1,0,1], 输出：3, 解释：删掉位置 2 的数后，[1,1,1] 包含 3 个 1 。
// 思路：滑动窗口 请看 sliding_window.h


// 5. 统计「优美数组」
// 题目描述：连续 子数组中恰好有 k 个奇数数字，我们就认为这个子数组是「优美子数组」。
// 思路：见sliding_window.h



// 6. 第一个只出现一次的字符
// 思路： 直观的解法是 遍历每一个元素，往后统计出现次数。只出现一次返回。O(n^2)
//   更好的解法是利用 map统计出现次数, O(N), 只需要两次扫描
//    当然我们可以用STL的unordered_map, 但基于这里是字符，ASCII字符范围只在[0,255],只需要开辟一个长256的数组统计出现次数
#define SIZE 256 //const int SIZE = 256
char appearOnlyOnceChar(string& s){
    vector<int> map(SIZE, 0);
    for(char ch: s){
        map[ch]++;
    }
    for(char ch: s){
        if(map[ch] == 1){
            return ch;
        }
    }
    return '\0';
}
void testappearOnlyOnceChar(){
    string s = "abbcadd";
    cout<<"char only appear once:"<< appearOnlyOnceChar(s)<<endl;
}

// 7. 数组中的逆序对
// 题目描述：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
// 思路：直观的解法是 遍历每个元素，依次比较和后面元素的大小，后面有小数就+1，复杂度O(n^2)
//  更好的解法是：归并排序思想, 不断划分子区间， 在 合并阶段统计逆序对，即需要交换的次数 复杂度O(nlogn)
void countReversePair(vector<int>& nums, int left, int mid, int right);
int mergeSort(vector<int>& nums, int left, int right);
int pairCount;
int reversePairs(vector<int>& nums) {
    // vector<int> temp(nums.size()); // temp[]为存放两组比较结果的临时数组，这里可以不需要
    return mergeSort(nums, 0, nums.size() - 1);
}
int mergeSort(vector<int>& nums, int left, int right) {
    int mid = left + ((right - left) >> 1);
    if (left < right) {
        mergeSort(nums, left, mid);
        mergeSort(nums, mid + 1, right);
        // 合并阶段统计逆序对，即需要交换的次数
        countReversePair(nums, left, mid, right); // 将mergeArray的过程替换为统计逆序对
    }
    return pairCount;
}
void countReversePair(vector<int>& nums, int left, int mid, int right) {
    vector<int> temp(right - left + 1); // temp[]为存放[left,mid] [mid+1,right]两子数组合并结果的临时数组
    int index = 0;
    int idx1 = left, idx2 = mid + 1;
    while (idx1 <= mid && idx2 <= right) {
        if (nums[idx1] <= nums[idx2]) { // 前面的数更小，是正序，可以将idx1填入temp
            temp[index++] = nums[idx1++];
        }
        else { // 否则，是逆序，统计个数，再idx2填入temp
            pairCount += (mid - idx1 + 1); // 当前逆序个数为(mid-idx1+1)， 因为[left,idx1]比他(idx2)小,而比他大的就是后面剩下这些
            temp[index++] = nums[idx2++];
        }
    }
    // 把左区间剩余的数移入数组，如果有的话
    while (idx1 <= mid) {
        temp[index++] = nums[idx1++];
    }
    // 把右区间剩余的数移入数组，如果有的话
    while (idx2 <= right) {
        temp[index++] = nums[idx2++];
    }
    // 把新数组中的数更新到nums数组
    for (int k = 0; k < temp.size(); k++) {
        nums[k + left] = temp[k];
    }
}
void testreversePairs(){
    int a[] = {7,3,2,6,0,1,5,4};
    vector<int> nums(a, a+ sizeof(a)/sizeof(a[0]));
    cout<<"pairs count = "<< reversePairs(nums)<<endl;
}
// ref: 最少交换次数使数组有序：https://blog.csdn.net/weixin_43664947/article/details/109137111
//

// 8. 绝对差值和
// 题目描述：数组 nums1 和 nums2 的 绝对差值和 定义为所有 |nums1[i] - nums2[i]|（0 <= i < n）的 总和（下标从 0 开始）。
//      你可以选用 nums1 中的 任意一个 元素来替换 nums1 中的 至多 一个元素，以 最小化 绝对差值和。
// 思路：使用 sum 记录所有的差值和，用 maxDiff 记录最大的改变前后的差值，这样答案即为 sum−maxDiff。
//   可以先对nums1进行排序然后二分查找nums1中最优的替换位置
const int mod = 1e9+7;
int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2) {
    long long sum = 0;
    set<long long> st;
    for(int i = 0; i < nums1.size(); ++i){
        sum += abs(nums1[i] - nums2[i]);
        st.insert(nums1[i]);
    }
    long long maxdiff = 0;
    for(int i = 0; i < nums2.size(); ++i){
        auto ptr = st.lower_bound(nums2[i]);
        if(ptr != st.end()){
            maxdiff = max(maxdiff, abs(nums1[i]-nums2[i]) - abs(nums2[i] - *ptr));
        }
        if(ptr != st.begin()){
            --ptr;
            maxdiff = max(maxdiff, abs(nums1[i]-nums2[i]) - abs(nums2[i] - *ptr));
        }
    }
    return (sum - maxdiff) % mod;
}



#endif //DATASTRUCT_ALGORITHM_ARRAY_STRING_OP_H

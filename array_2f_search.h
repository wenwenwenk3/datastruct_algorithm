//
// Created by kai.chen on 2021/12/23.
//  二分查找
//
//  1. 基本的二分查找:
//      查找指定值
//      搜索插入位置
//      查找左侧/右侧第一个出现的指定值
//  2.寻找旋转排序数组的最小值
//  4. 一次旋转的有序数组搜索指定值
//
//  5. 有效三角形的个数
//  6. 爱吃香蕉的珂珂
//  7. 找到最接近x的k个数

#ifndef DATASTRUCT_ALGORITHM_ARRAY_2F_SEARCH_H
#define DATASTRUCT_ALGORITHM_ARRAY_2F_SEARCH_H
#include <vector>
using namespace std;

//* 基本的二分查找：升序数组中查找指定的数，无重复数字
//需要注意的有两点，
//（1）尽量写成 mid=left+(right-left)/2 可以防止整型越界
//（2）left<=right区间表示相等的时候还有的搜。因为我们定义的right是闭区间。
int search(vector<int>& nums, int target) {
    int left=0, right = nums.size()-1;
    while(left<=right){
        int mid = left+(right-left)/2;
        if(nums[mid]==target){
            return mid;
        }
        else if(nums[mid]<target){
            left = mid+1;
        }
        else if(nums[mid]>target){
            right = mid-1;
        }
    }
    return -1;
}

//搜索要插入的位置：https://leetcode-cn.com/problems/search-insert-position/
int searchInsert(vector<int>& nums, int target) {
    int n = nums.size();
    if(n ==0) return 0;

    int left=0,right=n-1,pos=len; // 初始化插入位置为n
    while(left<=right){
        int mid = left+(right-left)/2;
        if(nums[mid] == target){
            pos = mid;
            break;
        }
        else if(target > nums[mid]){ // 往右区间搜
            left = mid+1;
        }
        else if(target < nums[mid]){
            pos = mid;
            right = mid-1;
        }
    }
    return pos;
}


//* 变体1：寻找左侧第一个出现的数字，或右侧。
// 若继续保持闭区间写法，
//  需注意检查left越界情况，因为while的退出条件时left==right+1
int left_bound_search(vector<int>& nums, int target) {
    int left=0, right = nums.size()-1;
    while(left<=right){
        int mid = left+(right-left)/2;
        if(nums[mid]==target){
            right = mid-1;
        }
        else if(nums[mid]<target){
            left = mid+1;
        }
        else if(nums[mid]>target){
            right = mid-1;
        }
    }
    if(left>=nums.size() || nums[left]!=target){
        return -1;
    }
    return left;
}
//* 右侧
// 同理，需要注意检查right越界情况
int right_bound_search(vector<int>& nums, int target) {
    if(nums.size()==0)
        return -1;
    int left=0, right = nums.size()-1;
    while(left<=right){
        int mid = left+(right-left)/2;
        if(nums[mid]==target){
            left = mid+1;
        }
        else if(nums[mid]<target){
            left = mid+1;
        }
        else if(nums[mid]>target){
            right = mid-1;
        }
    }
    if(right<0 || nums[right]!=target){
        return -1;
    }
    return right;
}

// 2.寻找旋转排序数组的最小值
// 思路：针对数组中最后一个元素x，在最小值左侧的元素一定都大于x，在最小值右侧都小于x
// 二分查找的每一步：左边界为l,右边界为r
int findMinValueInRotateArray(vector<int>& nums) { // [4,5,6,7,0,1,2]
    int len = nums.size();
    int l = 0, r = len-1;
    while(l < r){
        int mid = l + (r-l)/2;
        if(nums[mid] < nums[r]){ // 右边[mid+1,r] 是有序的部分,最小值在左边
            r = mid;
        }
        else{
            l = mid+1;
        }
    }
    return nums[l];
}

//* 4.旋转数组，已知排序数组经过了一次未知位置的旋转，同样搜索指定值 https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
// 思路：每次从数组中间任意位置分开两部分，一定有一部分是有序的
//    假设 [l,mid] 是有序的，当target满足 在[l,mid) 区间时, 缩短右区间：r = mid-1
int searchRotateArray(vector<int>& nums, int target) {
    int n = (int)nums.size();
    if (n == 0) return -1;
    if (n == 1) return nums[0] == target ? 0 : -1;
    int l = 0, r = n - 1;
    while (l <= r) {
        int mid = l + (r-l) / 2;
        if (nums[mid] == target) return mid;
        if (nums[l] <= nums[mid]) { // 通过和nums[l]比较就可确定左边是否有序，如果大于nums[l]则左区间有序
            if (nums[l] <= target && target < nums[mid]) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        } else {
            if (nums[mid] < target && target <= nums[r]) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
    }
    return -1;
}

// 5. 有效三角形的个数
//
// 排序+二分
int triangleNumber(vector<int> nums) {
    int n = nums.size();
    sort(nums.begin(), nums.end());
    int ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i - 1; j >= 0; j--) {
            int l = 0, r = j - 1;
            while (l < r) {
                int mid = l + r >> 1;
                if (nums[mid] + nums[j] > nums[i]) r = mid;
                else l = mid + 1;
            }
            if (l == r && nums[r] + nums[j] > nums[i]) ans += j - r;
        }
    }
    return ans;
} // N^2*logN

// 排序+双指针
//  注意到 当a=nums[i],b=nums[j] 时，最大的满足nums[k]<nums[i]+nums[j] 的下标 k 记为 ki,j
//  可以发现，如果我们固定 i，那么随着 j 的递增，不等式右侧 nums[i]+nums[j] 也是递增的，因此 ki,j也是递增的。
// 所以可以
//  - 使用一重循环枚举 i。当 i 固定时，我们使用双指针同时维护 j 和 k，它们的初始值均为 i；
//  - 每次将 j 向右移动一个位置，即 j←j+1，并尝试不断向右移动 k，使得 k 是最大的满足 nums[k]<nums[i]+nums[j] 的下标。然后只要将max(k−j,0) 累加入答案。
int triangleNumber(vector<int> nums) {
    int n = nums.size();
    sort(nums.begin(), nums.end());
    int ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i - 1, k = 0; k < j; j--) {
            while (k < j && nums[k] + nums[j] <= nums[i]) k++;
            ans += j - k;
        }
    }
    return ans;
} //


// 6. 爱吃香蕉的珂珂
// 题目描述：有 n 堆香蕉，返回珂珂可以在 h 小时内吃掉所有香蕉的最小速度 k（k 为整数）
// 思路：对速度进行二分，最小速度 左边都不满足，右边都满足
bool check(const vector<int>& piles, int k, int h) {
    long long hoursNeed = 0;
    for (const int& pile : piles){
        hoursNeed += ceil(pile * 1.0 / k);
    }
    return hoursNeed <= h;
}
int minEatingSpeed(vector<int>& piles, int h) {
    int l = 1, r = 0; // 这里r可以优化为piles的最大值，毕竟每堆香蕉最少也需要用一小时
    for_each(piles.begin(), piles.end(), [&r](const int& x){ r = max(r, x);});
    while (l < r) {
        int mid = l + (r-l)/2;
        if (check(piles, mid, h)) {
            r = mid;
        }else {
            l = mid+1;
        }
    }
    if(r<0 || !check(piles, r, h)){
        return -1;
    }
    return r;
} // O(N * logM) N是香蕉堆数，M是pile的最大值


//  7. 找到最接近x的k个数
// 原本的数组是有序的，所以我们可以像如下步骤利用这一特点。
// 思路：
//  (1)如果目标 x 小于等于有序数组的第一个元素，那么前 k 个元素就是答案。
//  (2)类似的，如果目标 x 大于等于有序数组的最后一个元素，那么最后 k 个元素就是答案。
//  (3)其他情况，我们可以使用二分查找来找到恰好大于 x 一点点的元素的索引 index 。
//   然后让 low 等于 index 左边 k-1 个位置的索引，high 等于 index 右边 k-1 个位置的索引。
//   我们需要的 k 个数字肯定在范围 [index-k-1, index+k-1] 里面。所以我们可以根据以下规则缩小范围以得到答案。
//      如果 low 小于 0 或者 low 对应的元素比 high 对应的元素更接近 x ，那么减小 high 索引。
//      如果 high 大于最后一个元素的索引 arr.size()-1 或者它比起 low 对应的元素更接近 x ，那么增加 low 索引。
//      当且仅当 [low, high] 之间恰好有 k 个元素，循环终止，此时范围内的数就是答案。
vector<int> findClosestElements(vector<int>& arr, int k, int x) {
    int left = 0;
    int right = arr.size() - k - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (x - arr[mid] > arr[mid + k] - x) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return vector<int>(arr.begin() + left, arr.begin() + left + k);
}


#endif //DATASTRUCT_ALGORITHM_ARRAY_2F_SEARCH_H

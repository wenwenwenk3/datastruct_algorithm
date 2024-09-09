//
// Created by kai.chen on 2021/12/16.
//
// 1. 搜索旋转排序数组
//  1.1 搜索旋转数组
//  1.2 旋转函数的最大值
// 2. 寻找旋转排序数组的最小值
//     延伸：旋转数组有重复数字寻找最小值II
// 3. 旋转一维数组
//  3.1 旋转链表
// 4. 旋转二维图像
//
// 5. 山脉数组中查找目标值


#ifndef DATASTRUCT_ALGORITHM_ARRAY_ROTATE_H
#define DATASTRUCT_ALGORITHM_ARRAY_ROTATE_H
#include <vector>

// 1.搜索旋转排序数组
// 思路：数组从中间分开成两部分，一定有一部分是有序的
int search(const vector<int>& nums, int target) { // [4,5,6,7,0,1,2]
    int len = nums.size();
    if(len == 0) return -1;
    if(len == 1) return nums[0] == target?0:-1;

    int left = 0, right = len-1;
    while(left <= right){
        int mid = left+(right-left)/2;
        if(nums[mid] == target){
            return mid;
        }
        else if(nums[0] <= nums[mid]){
            // 左边[left,mid-1] 是有序的部分
            if(nums[left]<=target && target <nums[mid]){
                right = mid-1; // 如果target在左边部分，往左边查找
            } else{
                left = mid+1;
            }
        }
        else if(nums[0] > nums[mid]){
            // 右边[mid+1,right] 是有序的部分
            if(nums[mid]<target && target <= nums[len-1]){
                left = mid+1; // 如果target在右边部分，往右边查找
            }else{
                right = mid-1;
            }
        }
    }

    return -1;
}

//  1.1 搜索旋转数组
// 题目描述：给定一个排序后的数组，包含n个整数，但这个数组已被旋转过很多次了，次数不详。https://leetcode.cn/problems/search-rotate-array-lcci/
//  请找出数组中的某个元素，若有多个相同元素，返回索引值最小的一个，没有找到返回-1 https://leetcode.cn/problems/search-rotate-array-lcci/solution/by-xiaowei_algorithm-0v63/


int maxRotateFunction(vector<int>& nums) {
    int f = 0, n = nums.size();
    int numSum = accumulate(nums.begin(), nums.end(), 0);
    for (int i = 0; i < n; i++) {
        f += i * nums[i];
    }
    int res = f;
    for (int i = n - 1; i > 0; i--) {
        f += numSum - n * nums[i];
        res = max(res, f);
    }
    return res;
}

// 2.寻找旋转排序数组的最小值
// 思路：针对数组中最后一个元素x，在最小值左侧的元素一定都大于x，在最小值右侧都小于x
// 二分查找的每一步：左边界为l,右边界为r
int findMin(vector<int>& nums) { // [4,5,6,7,0,1,2]
    int len = nums.size(); // [rollout]
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

// 延伸：旋转数组有重复数字寻找最小值II 【困难，有点意思】
// 思路：还是针对数组中的最后一个元素 x：在最小值右侧的元素，它们的值一定都小于等于 x；
//      由于重复元素的存在，我们并不能确定 nums[mid] 究竟在最小值的左侧还是右侧,因此我们不能莽撞地忽略某一部分的元素
//      当 nums[mid] == nums[right]时，无论 nums[right] 是不是最小值,都可以将right--，因为最少有一个替代品nums[mid]
int findMinII(vector<int>& nums) { // [4,5,6,7,0,1,4]
    int len = nums.size();
    int l = 0, r = len-1;
    while(l < r){
        int mid = l + (r-l)/2;
        if(nums[mid] < nums[r]){ // 右边[mid+1,r] 是有序的部分,最小值在左边
            r = mid;
        }
        else if(nums[mid] > nums[r]){
            l = mid+1;
        }
        else{ //  当 nums[mid] == nums[right]时，无论 nums[right] 是不是最小值,都可以将right--，因为最少有一个替代品nums[mid]
            r--;
        }
    }
    return nums[l];
}


// 3. 旋转数组
// 题目描述：给你一个数组，将数组中的元素向右轮转 k 个位置
// 方法一：使用额外的数组 直接存放每一个元素的正确位置
void rotate(vector<int>& nums, int k) {
    int n = nums.size();
    vector<int> newNums(n);
    for (int i = 0; i < n; ++i) {
        newNums[(i + k) % n] = nums[i];
    }
    nums.assign(newNums.begin(), newNums.end());
} // 时间复杂度O(N), 空间复杂度O(N)

// 方法二：先翻转所有元素，再反转[0，k-1]，然后反转[k, n-1]得到结果
void reverse(vector<int>& nums, int start, int end) {
    while (start < end) {
        swap(nums[start++], nums[end--]);
    }
}
void rotate_v2(vector<int>& nums, int k) {
    k %= nums.size();
    reverse(nums, 0, nums.size() - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, nums.size() - 1);
} // 时间复杂度O(N), 空间复杂度O(1)

// 3.1 旋转链表
// 题目描述： 将链表每个节点向右移动k个位置
//      输入：head = [1,2,3,4,5], k = 2
//      输出：[4,5,1,2,3]
// 思路：先将给定的链表连接成环，然后将指定位置断开
ListNode* rotateRight(ListNode* head, int k) {
    if (k == 0 || head == nullptr || head->next == nullptr) {
        return head;
    }
    int n = 1; // 统计节点个数
    ListNode* tail = head;
    for(; tail->next != nullptr; tail = tail->next) {
        n++;
    }
    int steps = n - k % n; //
    if (steps == n) return head;

    tail->next = head;

    while (steps--) {
        tail = tail->next;
    }
    ListNode* res = tail->next;
    tail->next = nullptr; // 断开tail->head循环
    return res;
}


// 4. 旋转二维图像
//  （推荐使用 array_2d 中的分圈处理解法，更容易理解）
// 题目描述：给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度
void rotate(vector<vector<int>>& matrix) {
    int n = matrix.size();
    // 水平翻转
    for (int i = 0; i < n / 2; ++i) {
        for (int j = 0; j < n; ++j) {
            swap(matrix[i][j], matrix[n - i - 1][j]);
        }
    }
    // 主对角线翻转
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            swap(matrix[i][j], matrix[j][i]);
        }
    }
} // 时间复杂度O(N^2),空间复杂度O(1)


// 5. 山脉数组中查找目标值
// 题目描述：请你返回能够使得mountainArr[index]等于target最小的下标 index值。
//      如果不存在这样的下标 index，就请返回-1。
//     输入：array = [1,2,3,4,5,3,1], target = 3， 输出：2
// 思路：类似于二分查找有序数组
//     假设山脉是完全递增或递减的，那明显可以直接使用二分法搜最左侧
//     现在情况是山脉峰值左侧是递增，右侧是递减。所以可以先来一次二分求出峰值
// 这道题讲清楚思路能写二分即可，我觉得不用掌握，解法过于复杂
class MountainArray{
private:
    vector<int> nums;
public:
    int get(int i){return nums[i];}
    int size(){ return nums.size();}
};
// 需要三次二分，所以先抽象一个二分查方法
int left_bound_search(MountainArray &nums, int target, int l, int r) {
    int left=l, right = r;
    while(left<=right){
        int mid = left+(right-left)/2;
        if(nums.get(mid)==target){
            right = mid-1;
        }
        else if(nums.get(mid)<target){
            left = mid+1;
        }
        else if(nums.get(mid)>target){
            right = mid-1;
        }
    }
    if(left>=nums.size() || nums.get(left)!=target){
        return -1;
    }
    return left;
}






#endif //DATASTRUCT_ALGORITHM_ARRAY_ROTATE_H

// 延伸，寻找旋转排序数组的最小值II，数组有重复元素
// https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/solution/xun-zhao-xuan-zhuan-pai-xu-shu-zu-zhong-de-zui--16/
//int findMin2(vector<int>& nums) {
//    int low = 0;
//    int high = nums.size() - 1;
//    while (low < high) {
//        int pivot = low + (high - low) / 2;
//        if (nums[pivot] < nums[high]) {
//            high = pivot;
//        }
//        else if (nums[pivot] > nums[high]) {
//            low = pivot + 1;
//        }
//        else {
//            high -= 1;
//        }
//    }
//    return nums[low];
//}

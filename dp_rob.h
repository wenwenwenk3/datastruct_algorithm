//
// Created by kai.chen on 2021/12/16.
//
//      1.打家劫舍
//      2.打家劫舍II (环形排列)
//      3.打家劫舍III(树形排列)

#ifndef DATASTRUCT_ALGORITHM_DP_ROB_H
#define DATASTRUCT_ALGORITHM_DP_ROB_H
#include <vector>
#include <unordered_map>
#include "binary_tree_normal_op.h"

// 1.打家劫舍
// 题目描述：nums表述一排房屋中钱数，约束不能同时取出相邻房子里的钱。
//      求最多取出多少钱？
// dp(nums, start)表示从nums[start]开始做选择，可以取得的最多的钱
int dp(vector<int>& nums, int start){
    if(start >= nums.size()){
        return 0;
    }
    // 对于当前房子，要么取要么不取
    int res = max(nums[start]+dp(nums, start+2), dp(nums, start+1));
    return res;
}
int rob(vector<int> nums){
    return dp(nums, 0);
}

// 优化1，明显存在重叠子问题可以用备忘录
// 优化2, 自底向上的思路
int rob_u2(vector<int> nums){
    int n = nums.size();
    vector<int> dp(n+2, 0);
    // 对于当前房子，要么取要么不取, base case是dp[n]=0
    for(int i = n-1; i >= 0; i--){
        dp[i] = max(dp[i+1], nums[i]+dp[i+2]);
    }
    return dp[0];
}
// 或：
// int rob(vector<int>& nums) {
//    int n = nums.size();
//    vector<int> f(n + 2, 0);
//    for (int i = 0; i < n; i++) {
//        f[i + 2] = max(f[i + 1], f[i] + nums[i]);
//    }
//    return f[n + 1];
//}

// 优化3, 上面的状态 可以状态压缩
int rob_u3(vector<int> nums){
    int n = nums.size();

    int dp_i_1 = 0, dp_i_2 = 0; // 备忘dp[i+1],dp[i+2]
    int dp_i;
    // 对于当前房子，要么取要么不取, base case是dp[n]=0
    for(int i = n-1; i >= 0; i--){
        dp_i = max(dp_i_1, nums[i] + dp_i_2);
        dp_i_2 = dp_i_1;
        dp_i_1 = dp_i;
    }
    return dp_i;
}


// 2.打家劫舍II (环形排列)
// 题目描述：房屋环形排列，同样的约束是不能再相邻的房子同时取钱
//      注意第一间房子和最后一间也相当于是相邻的
// 思路: 其实加的限制就是 首尾不能同时取钱
//   那要么取首不取尾， 要么取尾不取首，两者最大
// 改造上面的成rob_range来计算闭区间[l,r]取钱的最大值
int rob_range(vector<int> nums, int l, int r){ // [l,r]
    int n = nums.size();
    int dp_i_1 = 0, dp_i_2 = 0; // 备忘dp[i+1],dp[i+2]
    int dp_i;
    // 对于当前房子，要么取要么不取, base case是dp[n]=0
    for(int i = r; i >= l; i--){
        dp_i = max(dp_i_1, nums[i] + dp_i_2);
        dp_i_2 = dp_i_1;
        dp_i_1 = dp_i;
    }
    return dp_i;
}
int robII(vector<int>& nums){
    int n = nums.size();
    if(n == 1) return nums[0];
    // 要么取首不取尾， 要么取尾不取首
    return max(rob_range(nums, 0, n-2), rob_range(nums,1,n-1));
}


// 3.打家劫舍III(树形排列)
// 题目描述：房子在二叉树节点上，约束是不能同时取相连的两个节点
// 递归的每次做选择 要么取要么不取。取得话需要跳过两个直接相连的节点，不取的话直接往下
unordered_map<TreeNode*, int> memo; // 备忘每个节点的最优选择
int robIII(TreeNode* root){
    if(root == nullptr) return 0;
    if(memo.count(root)){
        return memo[root];
    }

    // 做选择，取或者不取
    int resIfDo = root->val
            + (root->left== nullptr ? 0: robIII(root->left->left)+ robIII(root->left->right))
            + (root->right== nullptr ? 0: robIII(root->right->right)+ robIII(root->right->left));
    int resIfNotDo = robIII(root->left)+robIII(root->right);

    int res = max(resIfDo, resIfNotDo);
    memo[root] = res;
    return res;
} // 时间复杂度是O(N),N为树的节点个数。空间为logN 递归栈的深度



#endif //DATASTRUCT_ALGORITHM_DP_ROB_H

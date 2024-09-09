//
// Created by kai.chen on 2022/10/23.
//

#ifndef DATASTRUCT_ALGORITHM_MATH_PROBLEM_F2_H
#define DATASTRUCT_ALGORITHM_MATH_PROBLEM_F2_H

// 1.有序数组中差绝对值之和
// 输入：nums = [2,3,5]
//输出：[4,3,5]
//解释：假设数组下标从 0 开始，那么
//result[0] = |2-2| + |2-3| + |2-5| = 0 + 1 + 3 = 4，
//result[1] = |3-2| + |3-3| + |3-5| = 1 + 0 + 2 = 3，
//result[2] = |5-2| + |5-3| + |5-5| = 3 + 2 + 0 = 5。
vector<int> getSumAbsoluteDifferences(vector<int>& nums) {
    int n = nums.size();
    vector<int> ans(n, 0);
    ans[0] = accumulate(nums.begin(), nums.end(), 0) - nums[0] * n;
    for (int i = 1; i < n; ++i) {
        int d = nums[i] - nums[i - 1];
        ans[i] = ans[i - 1] - (n - i * 2) * d;
    }
    return ans;
}




#endif //DATASTRUCT_ALGORITHM_MATH_PROBLEM_F2_H

//
// Created by kai.chen on 2022/6/12.
//
//  子集划分问题
//
//      1. 能否公平划分：火柴拼正方形473，划分为k个相等子集698
//              判断分割等和子集（背包思路）：https://leetcode.cn/problems/NUPfPr/solution/by-flix-c3fv/ [todo]
//                  https://leetcode.cn/problems/partition-equal-subset-sum/solution/0-1bei-bao-wen-ti-416-fen-ge-deng-he-zi-kb25t/
//      2. 尽量公平划分的最大值，公平分饼干5289 完成工作最短时间1723 分配重复数1655
//
//      3. 100 game 我能赢吗
//      4. 石子游戏 预测赢家
//
#ifndef DATASTRUCT_ALGORITHM_ARRAY_SUBSET_DIVIDE_H
#define DATASTRUCT_ALGORITHM_ARRAY_SUBSET_DIVIDE_H
#include <vector>
#include <numeric>
using namespace std;


// 1. 判断能否公平划分
// 思路:   (1) 回溯
//        (2) 状压dp
int K;
bool backtrack(vector<int>& matchsticks, int idx, vector<int>& edge, int edgetarget){
    if(idx == matchsticks.size()) return true;
    for(int i = 0; i < K; i++){
        edge[i] += matchsticks[idx];
        if(edge[i] <= edgetarget && backtrack(matchsticks, idx+1, edge, edgetarget)){
            return true;
        }
        edge[i] -= matchsticks[idx];
    }
    return false;
}
bool canPartitionKSubsets(vector<int>& matchsticks, int k) {
    int sum = accumulate(matchsticks.begin(), matchsticks.end(), 0);
    if(sum % k != 0) return false;
    K=k;

    sort(matchsticks.begin(), matchsticks.end(), greater<int>()); // 优化搜索速度，没有这步这题超时
    vector<int> edges(4);
    return backtrack(matchsticks, 0, edges, sum/K);
}// 时间复杂度k ^ n，空间复杂度n

// 状压dp
bool canPartitionKSubsets2(vector<int>& nums, int k) {
    // 数组长度小于k
    int n = nums.size();
    if (n < k) return false;
    int sum = accumulate(nums.begin(), nums.end(), 0);
    // 总和不是k的整数倍
    if (sum % k) return false;
    int target = sum / k;
    vector<int> dp(1 << n, -1);
    dp[0] = 0;
    for (int i = 1; i < (1 << n); i++) {
        for (int j = 0; j < n; j++) {
            if (i>> j & 1){ // 包含当前数字nums[j]
                int l = i & ~(1 << j);  // 除去这个数
                if (dp[l] >= 0 && dp[l] + nums[j] <= target) {
                    dp[i] = (dp[l] + nums[j]) % target;
                }
            }
        }
    }
    return dp[(1 << n) - 1] == 0;
} // 时间复杂度n * 2 ^ n，空间复杂度2 ^ n

// k=2
bool canPartition(vector<int> nums) {
    int len = nums.size();
    int sum = 0;
    for (int num : nums) {
        sum += num;
    }
    if ((sum & 1) == 1) {
        return false;
    }

    int target = sum / 2;
    vector<bool> dp (target + 1);
    dp[0] = true;

    if (nums[0] <= target) {
        dp[nums[0]] = true;
    }
    for (int i = 1; i < len; i++) {
        for (int j = target; nums[i] <= j; j--) {
            if (dp[target]) {
                return true;
            }
            dp[j] = dp[j] || dp[j - nums[i]];
        }
    }
    return dp[target];
}



// 2. 尽量公平的分饼干后的最大值
// 题目描述：
// 思路： (1) 回溯搜索:  O(k^n) 每个饼干都可以有k种选择,相当于k叉树，树的高度为n
//       (2) 二分+剪枝: 满足二段性，求满足条件的最左侧值
//       (3) 状压dp O(n*3^n)
// 回溯搜索
vector<int> cookies;
vector<int> divided;
int k;
static int _res_cookie = 0x3f;
//  当前在分cookies[curidx]袋饼干
void backtrackcook(int curidx){
    if(curidx == cookies.size()){
        int maxCookies = 0;
        for_each(divided.begin(), divided.end(), [&maxCookies](int x){
            maxCookies = max(maxCookies, x); // 当然这里可以优化，添加一个跟着更新的max状态变量就行。这里为了容易理解
        });
        _res_cookie = min(_res_cookie, maxCookies);
        return ;
    }
    for(int i = 0; i < k; i++){
        divided[i] += cookies[curidx]; // 做选择
        backtrackcook(curidx+1);
        divided[i] -= cookies[curidx]; // 撤销选择
    }
}
int distributeCookies1(vector<int>& _cookies, int _k) {
    cookies = _cookies;
    k = _k;
    divided.resize(k, 0);
    backtrackcook(0);
    return _res_cookie;
}

// 回溯+二分+剪枝
vector<int> Jobs;
int sumJobs;
int limit;
// 当前在分Jobs[curidx]任务，限制每个人最大为limit
bool backtrackjob(int curidx){
    if(curidx == Jobs.size()){ // 成功分完
        return true;
    }
    for(int i = 0; i < k; i++){
        divided[i] += Jobs[curidx]; // 分给第i个人
        if(divided[i] <= limit){
            if(backtrackjob(curidx+1)) { // 递归分成功
                return true;
            }
        }
        // 能走到这已经可以说明Jobs[curidx]分给第i个人失败
        divided[i] -= Jobs[curidx]; // 撤销

        // 如果第i个人的工作量本来就为0的 都无法分配成功，那可以直接退出，Jobs[curidx]往后面分给谁都不可能成功
        if(divided[i] == 0) return false;
    }
    return false;
}
// 检查把jobs分给k个人，最大值不超过max_target是否可能
bool checkValidDivide(vector<int>& jobs, int _k, int max_target){
    int sum = accumulate(jobs.begin(), jobs.end(), 0);
    if(max_target * _k < sum) return false;
    limit = max_target;
    for_each(divided.begin(), divided.end(), [](int& x){ x=0;}); // 重置分配结果
    return backtrackjob(0);
}
int minimumTimeRequired(vector<int>& jobs, int _k) {
    Jobs = jobs;
    k = _k;
    divided.resize(k,0);
    sumJobs = accumulate(Jobs.begin(), Jobs.end(), 0);
    sort(Jobs.begin(), Jobs.end(), greater<int>()); // 从大工作量的任务开始枚举更快触发失败
    int l = max(Jobs[0], sumJobs/k);
    int r = sumJobs;
    while(l < r){
        int mid = l+(r-l)/2;
        if(checkValidDivide(Jobs, _k, mid)){
            r = mid;
        }else{
            l = mid+1;
        }
    }
    return l;
} // 时间复杂度: O(n*logn + log(r-l) * k^n), 实际搜索部分因为各种剪枝远远到不到 k^n
// 空间复杂度: O(n), 递归栈的深度


// 状态压缩 dp
//  答案和输入的顺序无关，
//  有消耗的概念，集合的划分, 数据的个数比较小
// dp[i][j] 表示给前 i 个人分配工作，工作的分配情况为 j (由二进制位表达) 时，完成所有工作的最短时间。
//   dp[i][j] = min( dp[i-1][]
int minimumTimeRequired3(vector<int>& jobs, int k) {
    int n = jobs.size();
    vector<int> sum(1 << n);
    for (int i = 1; i < (1 << n); i++) {
        for(int j = 0; j < n; j++){
            if(i>> j & 1){
                sum[i] += jobs[j];
            }
        }
    }

    // base case
    vector<vector<int>> dp(k, vector<int>(1 << n));
    for (int j = 0; j < (1 << n); j++) {
        dp[0][j] = sum[j];
    }

    for (int i = 1; i < k; i++) {
        for (int j = 0; j < (1 << n); j++) {
            int minn = INT_MAX;
            for (int x = j; x; x = (x - 1) & j) { // 枚举子集
                minn = min(minn, max(dp[i - 1][j - x], sum[x]));
            }
            dp[i][j] = minn;
        }
    }
    return dp[k - 1][(1 << n) - 1];
}
// 状压+滚动数组
int distributeCookies(vector<int>& cookies, int k){
    int n = cookies.size();
    vector<int> sum(1<<n); // 2^n
    for(int i = 1; i < 1<<n; i++){
        for(int j = 0; j < n; j++){
            if(i>> j & 1){
                sum[i] += cookies[j];
            }
        }
    }
    vector<int> f(sum);
    for(int i = 1; i < k; i++){
        for(int j = (1<<n)-1; j>0; j--){
            for(int s = j; s ; s = (s-1) & j){
                f[j] = min(f[j], max(f[j^s], sum[s]));
            }
        }
    }

    return f.back();
}//时间复杂度: O(n*3^n)

void test_distributeCookies(){
    vector<int> nums = {4, 3, 2, 3, 5, 2, 1};
    int k = 4;
    cout << canPartitionKSubsets2(nums, k) <<endl;

    vector<int> cookies1 = {8,15,10,20,8};
    int k1 = 2; //

    vector<int> jobs = {1,2,4,7,8};
    cout << minimumTimeRequired(jobs, k1)<<endl;

}

// 3. 100 game 我能赢吗
// 题目描述：在 "100 game" 这个游戏中，两名玩家轮流选择从 1 到 10 的任意整数，累计整数和，先使得累计整数和 达到或超过  100 的玩家，即为胜者
//      现在将游戏规则改为 “玩家 不能 重复使用整数”，即不放回 (1<=maxChoosableInteger<=20, 0<=desiredTotal<=300)
// 思路：
//     (1) 加备忘录的状压搜索
//     (2) 状压dp
vector<int> memo_100game;
int desiredTotal_100game;
// usedNumbers标记每个数字是否用过
int dfs_100game(int maxChoosableInteger, int usedNumbers, int curTotal) {
    if (memo_100game[usedNumbers] != 0){
        return memo_100game[usedNumbers];
    }

    for (int i = 1; i <= maxChoosableInteger; i++) {
        // 选i 能赢的必须满足条件： (1)数字i没用过 (2)选i直接能赢 或 选了i之后对方无法赢
        if (((usedNumbers >> (i-1)) & 1) == 0 &&
                ((i + curTotal >= desiredTotal_100game) ||
                (dfs_100game(maxChoosableInteger, usedNumbers | (1 << (i-1)), curTotal + i ) == -1)))
        {
            memo_100game[usedNumbers] = 1;
            return 1;
        }
    }
    // 到这表示不能赢
    memo_100game[usedNumbers] = -1;
    return -1;
}
// 给定两个整数 maxChoosableInteger （整数池中可选择的最大数）和 desiredTotal（累计和）
bool canIWin(int maxChoosableInteger, int desiredTotal) {
    // 当所有数字选完仍无法到达 desiredTotal 时，两人都无法获胜，返回 false
    if ((1 + maxChoosableInteger) * (maxChoosableInteger) / 2 < desiredTotal) {
        return false;
    }
    desiredTotal_100game = desiredTotal;
    memo_100game.resize(1<<maxChoosableInteger, 0); // 1表示能赢，-1表示不能赢。0表示还未计算。预留0位置不用
    return dfs_100game(maxChoosableInteger, 0, 0) == 1;
} // 时间复杂度：O(2^n×n) 其中n=maxChoosableInteger，备忘录中dfs最多2^n次，每次O(n)时间
// 空间复杂度：O(2^n)

void testcanIWin(){
    canIWin(10,0);
}


// 4. 预测赢家
// https://leetcode.cn/problems/predict-the-winner/solution/yu-ce-ying-jia-by-leetcode-solution/


#endif //DATASTRUCT_ALGORITHM_ARRAY_SUBSET_DIVIDE_H

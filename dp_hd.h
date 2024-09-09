//
// Created by kai.chen on 2021/12/16.
//
// 1. 高楼扔鸡蛋
// 2. 0-1背包问题，
//      2.1. 背包最多装的价值
//      2.2. 一和零的最大子集
//      2.3. 零钱兑换、零钱兑换II
//
//          注：更多请见三叶小姐姐背包串烧(慎点)：https://leetcode-cn.com/problems/coin-change-2/solution/gong-shui-san-xie-xiang-jie-wan-quan-bei-6hxv/
//                                      https://leetcode.cn/problems/perfect-squares/solution/gong-shui-san-xie-xiang-jie-wan-quan-bei-nqes/
// 3. 戳气球
// 4. 四键键盘
//  . 零钱兑换 在2.3
//
// 5. 交错字符串
//
// 6. 完全平方数
// 7. 统计不同回文子序列

#ifndef DATASTRUCT_ALGORITHM_DP_HD_H
#define DATASTRUCT_ALGORITHM_DP_HD_H
#include <vector>
#include <unordered_map>
using namespace std;

// 1. 高楼扔鸡蛋
// 题目描述：1-N 一共N层楼，K个鸡蛋(K>=1),问最坏情况需要几次找出楼层F(0=< F <=N)扔鸡蛋恰好没碎
// （注意，鸡蛋碎了就不能再用了，没碎还能捡回来继续扔）
//
// 思考思路：所以： 最坏情况是，从1楼开始扔，一直扔到N楼都没碎，需要N次
//   当然可以二分搜索策略，从（1+7）/2 = 4开始扔，
//      如果碎了说明F小于4，从（1+3）/2 = 2开始扔
//      如果没碎说明F大于4，从（5+7）/2 = 6开始扔
//   这种策略最坏情况是是到F=7/0还没碎，只需要log7次找出F
//////实际上，如果不限制鸡蛋的个数，这种方式显然可以得到最少的尝试次数。
//
// 动归思路："状态"是鸡蛋个数K，剩余测试楼层N。 "选择"就是从哪层扔鸡蛋
//  从第i层扔鸡蛋，如果碎了：K=K-1, 测试楼层从[1..N]变成[1..i-1]共i-1层
//          如果没碎：K不变，测试楼层从[1..N]变成[i+1..N]共N-i层 /不用取i是因为，F可以等于0，相当于0
// base case 是N==0 不用扔，和K==1只能一层一层扔N次
unordered_map<string, int> memo;
int superEggDrop(int K, int N){
    if(N == 0) return 0;
    if(K == 1) return N;

    string key = to_string(K) + "," + to_string(N);
    if(memo.count(key)) return memo[key]; // 消除重复计算

    int res = INT_MIN;
    for(int i=1; i <= N; i++){
        res = min(res, max(superEggDrop(K, N-i), superEggDrop(K-1, i-1)) + 1);
    }
    memo[key] = res;

    return res;
} // 时间复杂度：子问题的个数 * 函数本身的复杂度 => KN * N


// 2. 0-1背包问题
// 题目描述：重量为W的背包和N个物品，第i个物品的重量和价值为wt[i],val[i]
//      求这个背包最多能装的价值
// 动归思路： "状态"：背包的容量和可选择的物品
//          "选择"：装还是不装
//   dp定义：dp[i][w]表示对于前i个物品，当前背包的容量为w可以装的最大价值
// 最终要求的答案就是dp[N][W]
int knapsack(int W, int N, vector<int>& wt, vector<int>& val){
    vector<vector<int>> dp(N+1, vector<int>(W+1, 0));

    for(int i = 1; i <= N; i++){ // i从1开始，第i个物品
        for(int w = 1; w <= W; w++){
            if(w - wt[i-1] < 0){
                // 注意背包容量不够，只能不装i
                dp[i][w] = dp[i-1][w];
            }
            else{
                // 容量够，选择装i或者不装i的最大值
                dp[i][w] = max(dp[i-1][w-wt[i-1]] + val[i-1],
                    dp[i-1][w]);
            }
        }
    }
    return dp[N][W];
}


// 2.2 一和零的最大子集
//输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
//输出：4
//解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
//其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。
//
// 思路：01背包的升级，背包只有最重量一个限制，这道题有两个限制，即选取的字符串子集中的 0 和 1 的数量上限
//    经典01背包 是用二维dp求解，这里可以用三维
//  动归思路：
//      dp定义： dp[i][j][k] 表示在前i 个字符串中，使用 j 个 0 和 k 个 1 的情况下最多可以得到的字符串数量。
//          假设数组str 的长度为 l，则最终答案为 dp[l][m][n]。
//
//      base case:
//          i=0 时，dp[i][j][k] = 0
//  时间复杂度：O(lmn + L) L是所有字符串长度和 动态规划需要计算的状态总数是 O(lmn)，每个状态的值需要 O(1)的时间计算。
//          对于数组strs 中的每个字符串，都要遍历字符串得到其中的 0 和 1 的数量，因此需要 O(L) 的时间遍历所有的字符串
//  空间复杂度：O(lmn)
vector<int> get01Count(string& str) {
    vector<int> res(2,0); // res[0]表示0的数量，res[1]表示1的数量，
    for (char ch : str) {
        res[ch - '0']++;
    }
    return res;
}

int findMaxForm(vector<string>& strs, int m, int n) {
    int length = strs.size();
    // 求的答案是dp[l][m][n]，创建对于的dp三维数组
    vector<vector<vector<int>>> dp(length + 1, vector<vector<int>>(m + 1, vector<int>(n + 1)));

    for (int i = 1; i <= length; i++) {
        vector<int> arr01 = get01Count(strs[i - 1]);
        int count0 = arr01[0], count1 = arr01[1];

        for (int j = 0; j <= m; j++) {
            for (int k = 0; k <= n; k++) {
                // 如果count0 > j 或 count1 > k时，不能选择当前字符串i
                if(count0 > j || count1 > k){
                    dp[i][j][k] = dp[i - 1][j][k];
                }
                // 否则，要么选择当前字符串i，要么不选当前字符串i
                else {
                    dp[i][j][k] = max(dp[i - 1][j][k], dp[i - 1][j - count0][k - count1] + 1);
                }
            }
        }
    }
    return dp[length][m][n];
}

// 2.3 零钱兑换
// 题目描述：求凑成总金额所需的 最少的硬币个数
//   例如： coins = [1,2,5],amount=11  输出：3 （5+5+1=11）
// 思路：
//  定义 dp(i) 为组成金额 i 所需最少的硬币数量
//   假设最后一枚面值为 c, 那么 dp[i] = dp[i-c] + 1; 而最后一枚面值 c 有可能为任意一个，需要找一个c让dp[i-c]最小
int coinChange(vector<int>& coins, int amount) {
    // 输入保证coins数组长度>1 且 coins[i]>=1
    vector<int> dp(amount + 1, INT_MAX);
    // base case
    dp[0] = 0;

    // dp[i] = min(dp[i-c]) + 1
    for (int i = 1; i <= amount; i++) {
        int tempMin = INT_MAX;
        for (int j = 0; j < coins.size(); j++) {
            if (coins[j] <= i) { // 硬币面值要 <= i
                tempMin = min(tempMin, dp[i - coins[j]]);
            }
        }

        dp[i] = tempMin == INT_MAX? INT_MAX: tempMin + 1;
    }
    return dp[amount] == INT_MAX ? -1 : dp[amount];
}// 时间复杂度：O(amount×n)

// 用最少的尾数为k的数 凑目标和
const int inf = 0x3f3f3f3f;
class Solution_kCou {
public:
    int minimumNumbers(int num, int k) {
        vector<int> dp(num + 1, inf);
        dp[0] = 0;
        for(int i=0; i<=num; ++i){
            if(dp[i] == inf)    continue;
            for(int j=1; i+j<=num; ++j){
                if(j % 10 != k) continue;
                dp[i+j] = min(dp[i+j], dp[i] + 1);
            }
        }
        if(dp[num] == inf)  return -1;
        return dp[num];
    }
}; // 时间复杂度：O(num*C)

// 2.3 零钱兑换II
// 题目描述：求可凑成总金额的不同 组合 个数
// 思路：dp, dfs搜索
//  类比 爬楼梯是求达到目标值的不同 排列数，例如target=3，[1,2]和[2,1]是两个答案
//  而这里 找零无非就是达到目标值的不同 组合数，[1,2]和[2,1]是一个答案需要去重
// 爬楼梯的dp 转移方程：dp[i] = cigma(dp[i-x])  (x in steps)
// 换零钱的dp 转移方程：dp[k,i] = dp[k-1,i] + dp[k, i-k] 「既然要考虑去重，那就增加一个维度一个一个硬币来选」
//   定义DP[k][i], 为前 k 个硬币凑齐金额 i 的组合数。最终求dp[coins.size()][amount]
//   解释：前k个硬币能组成target的方法数 = 前k-1 个硬币就能组成target 的方法数 + 前k 能组成 target-k 的方法数 （即用第k个硬币和不用第k个硬币方法数之和）
int change_dpv1(int amount, vector<int>& coins) {
    int K = coins.size() + 1;
    vector<vector<int>> dp(K, vector<int>(amount+1, 0));

    // base case: 前k个金币能凑成 0的方法数为1，这里主要是控制 搜索到0了就需要成功将方法数加一
    for (int k = 0; k <= coins.size(); k++){
        dp[k][0] = 1;
    }
    //
    for (int k = 1; k <= coins.size() ; k++){
        for (int i = 1; i <= amount; i++){
            if (i-coins[k-1] >= 0) {
                dp[k][i] = dp[k][i-coins[k-1]] + dp[k-1][i];
            } else{
                dp[k][i] = dp[k-1][k];
            }
        }
    }
    return dp[coins.size()][amount];
}// 时间复杂度：O(amount * coins.size() ), 总共需要填 K * amount的二维dp数组
// 空间复杂度：O(amount * K)

// 第二种dp思路： 定义 dp[i] 表示金额之和等于 i 的硬币组合数
int change_dpv2(int amount, vector<int>& coins) {
    vector<int> dp(amount+1, 0);
    dp[0] = 1;
    for (int coin : coins){ // 枚举硬币， 用前k个硬币能
        for (int i = 1; i <= amount; i++){ //枚举金额
            if (i - coin >= 0){ // 当前coin不能大于当前要凑的总数
                dp[i] += dp[i-coin];
            }
        }
    }
    return dp[amount];
} // 时间复杂度：O(amount * coins.size() ), 对于每个coin币面值都需要 遍历更新 amount长度的 dp数组
// 空间复杂度：O(amount)

// 简化代码
// 第一层循环：针对每一个硬币
// 第二层循环：需要更新比这个硬币面额大的数额
// 新的方法数 = 不采用这个硬币的方法数（之前的） + 采用这个硬币的方法数
int change(int amount, vector<int>& coins) {
    vector<int> dp(amount + 1);
    // base case: dp[0] = 1 (只有当不选取任何硬币时，金额之和才为 0，因此只有 1 种硬币组合)
    dp[0] = 1;
    // 第一层循环：针对每一个硬币
    // 第二层循环：需要更新比这个硬币面额大的数额
    // 新的方法数 = 不采用这个硬币的方法数（之前的） + 采用这个硬币的方法数
    for (int& coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] += dp[i - coin];
        }
    }
    return dp[amount];
}

// 已知，硬币面额[1,2,5,10]，求凑成总金额money的不同组合
// dfs搜索
vector<vector<int>> resofCoinChange;
void coinChange(int money, vector<int>& coins, int lastSum, int lastCoin, vector<int>& track){
    if(lastSum == money){
        resofCoinChange.push_back(track);
        return ;
    }
    for(const auto& coin : coins){
        if(lastSum < money && coin >= lastCoin){ // 每次coin要大于等于lastCoin防止回
            track.push_back(coin);
            coinChange(money, coins, lastSum+coin, coin, track);
            track.pop_back();
        }
    }
} // 时间复杂度: O(n^s) 是硬币数组大小，s是目标金额，因为n叉树的最大深度不会超过s
// 空间复杂度：O(s), 递归的层数

void testcoinChange(){
    vector<int> coins = {2};
    int amount = 3;
    vector<int> path;
    coinChange(amount, coins,0, 0, path);
}


// 3. 戳气球
// 题目描述：nums数字表示一排带有分数的气球，求戳破所有气球最多可能获得多少分
//  积分规则是：当戳破气球i, 获得的分数是nums[left]*nums[i]*nums[right]
//  （注意，left不一定就是i-1，nums[-1]和nums[len(nums)]是两个虚拟气球，分数为1）
// 常规思路：
//    （穷举：回溯的暴力穷举，动归的递推穷举）
//    回溯：把它当成穷举的话，不就是全排列问题，但时间复杂度是阶乘级的，肯定不能满足要求
// 动归思路：
//    这题难点在，动归要求子问题必须独立。而这里是有相关性的
//  首先将 -1和n加进去，这样数组长度为n+2，索引范围为[1，n].问题转化为戳破所有中间的气球，只剩下首尾气球后的最大分数
//
//  dp定义: dp[i][j]表示戳破气球(i,j)之间的所有气球的最高分数，最终要求的结果是dp[0][n+1]
//      base case: 当j<=i+1时没有气球，dp[i][j] = 0
//     转移过程： 不妨假设(i,j)之间最后戳破的球为k，就是选择一个k让分数最大
//        for( k=[i+1,j) )  dp[i][j]=max(dp[i][j], dp[i][k] + dp[k][j] + points[i]*points[k]*points[j]);
//     那么怎么遍历ij区间呢？basecase是矩形的左下角，dest是右上角，自然可以从下往上，从左往右填。
//void backtrack(vector<int>& nums, int score){
//    if(nums.empty()){
//        res = max(res, score);
//        return ;
//    }
//    for(int i = 0; i < nums.size(); i++){
//        int point = nums[i-1] * nums[i] * nums[i+1];
//        int temp = nums[i];
//        // 做选择
//        // 在nums中删除元素nums[i]
//        backtrack(nums, score+point);
//        // 取消选择
//        // 还原temp到nums[i]
//    }
//}

// 从下往上，从左往右的写法
int maxCoins(vector<int>& nums){
    int n = nums.size();
    // 添加虚拟气球
    vector<int> points(n+2,0);
    points[0] = points[n+1] = 1;
    for(int i = 0; i < n; i++){
        points[i+1] = nums[i];
    }
    // dp数组
    vector<vector<int>> dp(n+2, vector<int>(n+2,0));
    // i从下往上
    for(int i = n; i >= 0; i--){ // 本来i=[0,n+1] 从n开始是因为斜对角的n+1已经填了0
        // j从左往右
        for(int j = i+1; j < n+2; j++){
            // 做选择k
            for(int k = i+1; k < j; k++){
                dp[i][j]=max(dp[i][j], dp[i][k] + dp[k][j] + points[i]*points[k]*points[j]);
            }
        }
    }
    return dp[0][n+1];
}



// 4. 四键键盘
// 题目描述：四键:A,ctrl+a,ctrl+c,ctrl+v, 问如何在 N 次敲击按钮后得到最多的 A
// 思路：（这是一道典型的不同的dp状态定义思路导致不同的计算思路和效率）
//      状态定义1，一是剩余的按键次数n，当前屏幕上字符 A 的数量num，剪切板中字符 A 的数量copy
//    base case: n=0时，num就是答案
//    选择（转移过程）：
//    dp(n - 1, a_num + 1, copy),    # A
//      解释：按下 A 键，屏幕上加一个字符
//      同时消耗 1 个操作数
//
//    dp(n - 1, a_num + copy, copy), # Ctrl-V
//      解释：按下 C-V 粘贴，剪切板中的字符加入屏幕
//      同时消耗 1 个操作数
//
//    dp(n - 2, a_num, a_num)        # Ctrl-A Ctrl-C
//      解释：全选和复制必然是联合使用的，
//      剪切板中 A 的数量变为屏幕上 A 的数量
//      同时消耗 2 个操作数

// 时间复杂度，所有的状态总数即N^3，即使使用memo备忘录的方式也需要

// 状态定义2，「选择」还是那 4 个，但是这次我们只定义一个「状态」，也就是剩余的敲击次数 n。
//  这个算法基于这样一个事实，最优按键序列一定只有两种情况：
//
//      要么一直按 A：A,A,...A（当 N 比较小时）。 因为N较小时，C-A C-C C-V 这一套操作的代价相对比较高，可能不如一个个按 A
//
//      要么是这么一个形式：A,A,...C-A,C-C,C-V,C-V,...C-V（当 N 比较大时）。因为而当 N 比较大时，后期 C-V 的收获肯定很大
//
//  定义：dp[i] 表示 i 次操作后最多能显示多少个 A（基于上面的分析最后一次按键要么是 A 要么是 C-V）
//  for (int i = 0; i <= N; i++)
//    dp[i] = max(这次按 A 键，这次按 C-V)
//      => 按 A 键，就比上次多一个 A 而已: dp[i] = dp[i - 1] + 1;
//      => 按 C-V键，基于上面的分析最优的操作序列一定是 C-A C-C 接着若干 C-V，所以我们用一个变量 j 作为若干 C-V 的起点。那么 j 之前的 2 个操作就应该是 C-A C-C 了：
int maxA(int N) {
    vector<int> dp(N + 1);
    dp[0] = 0;
    for (int i = 1; i <= N; i++) {
        // 按 A 键
        dp[i] = dp[i - 1] + 1;
        for (int j = 2; j < i; j++) {
            // 全选 & 复制 dp[j-2]，连续粘贴 i - j 次
            // 屏幕上共 dp[j - 2] * (i - j + 1) 个 A
            dp[i] = max(dp[i], dp[j - 2] * (i - j + 1));
        }
    }
    // N 次按键之后最多有几个 A
    return dp[N];
} // 时间复杂度 O(N^2)，空间复杂度 O(N)



// 5. 交错字符串
// 思路：第一反应是双指针 去做，但这是错误的！
//  例如：s1 = "aabbcc" s2 = "dbbca" s3 = "aadbbcbcac"
//      双指针的结果是false，而实际是true
// 正确的思路是dp：
//     dp[i][j] 表示 s1 前i个字符与 s2 前 j 个字符能否交错形成 s3 的前 i+j 个字符
//  接下来：思考什么时候能形成呢？
//   情况(1)： s1 的前 i-1 个字符加 s2 的前 j 个字符能匹配 s3 的前 i+j-1 个字符，且 s1 的第 i 个字符等于 s3 的第 i+j 个字符
//   情况(2)： s1 的前 i 个字符加 s2 的前 j-1 个字符能匹配 s3 的前 i+j-1 个字符, 且 s2 的第 j 个字符等于 s3 的第 i+j 个字符
// base case:  dp[0][0] == true
int isInterleave(const string& s1, const string& s2, const string& s3){
    int m = s1.size(), n = s2.size();
    if(m + n != s3.size()) return false;
    vector<vector<bool>> dp(m+1, vector<bool>(n+1, false));
    dp[0][0] = true;
    for(int i = 0; i <= m; i++){
        for(int j = 0; j <=n; j++){
            if(j-1 >= 0){ // 注意数组越界
                dp[i][j] = dp[i][j] || (dp[i][j-1] && s2[j-1] == s3[i+j-1]);
            }
            if(i-1 >= 0){
                dp[i][j] = dp[i][j] || (dp[i-1][j] && s1[i-1] == s3[i+j-1]);
            }
        }
    }
    return dp[m][n];
}

// 6. 完全平方数 https://leetcode.cn/problems/perfect-squares/solution/gong-shui-san-xie-xiang-jie-wan-quan-bei-nqes/
// dp[n] 表示最少需要多少个数的平方来表示整数 n。
// dp[n] = 1+min{j=1,n^1/2}dp[n-j*j]
int numSquares(int n) {
    vector<int> dp(n + 1);
    for (int i = 1; i <= n; i++) {
        int minn = INT_MAX;
        for (int j = 1; j * j <= i; j++) {
            minn = min(minn, dp[i - j * j]);
        }
        dp[i] = minn + 1;
    }
    return dp[n];
}


#endif //DATASTRUCT_ALGORITHM_DP_HD_H

//
// Created by kai.chen on 2022/6/2.
//
//  买卖股票问题
//      1. 买卖股票的最佳时机（easy）限定交易次数 k=1 也称股票的最大利润
//      2. 买卖股票的最佳时机 II（medium）交易次数无限制 k = +infinity
//      3. 买卖股票的最佳时机 III (hard) 限定交易次数 k=2
//      4. 买卖股票的最佳时机 IV (hard) 限定交易次数 最多次数为 k
//      5. 最佳买卖股票时机含冷冻期(medium) 含有交易冷冻期
//      6. 买卖股票的最佳时机含手续费 (medium) 每次交易含手续费
//

#ifndef DATASTRUCT_ALGORITHM_DP_BUY_STOCKS_H
#define DATASTRUCT_ALGORITHM_DP_BUY_STOCKS_H
#include <vector>
using namespace std;

// 1.买卖股票的最佳时机（也称股票的最大利润）
//  题目描述：数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。返回你可以从这笔交易中获取的最大利润
//  暴力求解每组 i 和 j（其中j>i）我们需要找出max(prices[j]−prices[i])。
//      时间复杂度 O(n^2)
int maxProfit(vector<int>& prices) {
    int n = (int)prices.size(), ans = 0;
    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; ++j) {
            ans = max(ans, prices[j] - prices[i]);
        }
    }
    return ans;
}
// 方法2 一次遍历 [7, 1, 5, 3, 6, 4]
// 只需要控制在历史最低点买入，价格为minPrice, 而在第i天卖出得到的利润就是price[i] - minPrice
// 然后每一天只需要考虑这么一个问题：如果我是在历史最低点买进的，那么我今天卖出能赚多少钱？
int maxProfit(vector<int>& prices) {
    if(prices.size()< 1) return 0;
    int minV = INT_MAX;
    int res = 0;
    for(int i = 0; i < prices.size()-1; i++){ // 在i+1卖出
        minV = min(minV, prices[i]); // [0:i]区间的最小值
        res = max(res, prices[i+1]-minV);
    }
    return res;
} //
int maxProfit2(vector<int>& prices) {
    int minprice = INT_MAX, maxprofit = 0;
    for(int i = 0; i < prices.size(); i++) {
        maxprofit = max(maxprofit, prices[i] - minprice);
        minprice = min(prices[i], minprice);
    }
    return maxprofit;
}

// 2.买卖股票的最佳时机 II
// 题目描述：设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）
//      注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
//          输入: prices = [7,1,5,3,6,4]
//          输出: 7
//          解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
//              随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
// 思路：相当于求递增序列的差值，因为不能同时参与多笔交易，因此每天交易结束后只可能存在手里有一支股票或者没有股票的状态。
//      定义定义状态dp[i][0] 表示第 i 天交易完后手里没有股票（也就是第i天卖出）的最大利润，
//          dp[i][1] 表示第 i 天交易完后手里持有一支股票（也就是第i天不卖）的最大利润（i 从 0 开始）。
//      dp[i][0]=max{dp[i−1][0],dp[i−1][1]+prices[i]}
//      => max(dp[i−1][0]:前一天已经没有股票, dp[i−1][0]:前一天结束的时候手里持有一支股票+第i天卖出的价格)
//      dp[i][1]=max{dp[i−1][1],dp[i−1][0]−prices[i]}
//      => max(dp[i−1][1]:前一天已经持有一支股票并且没卖, dp[i−1][0]−prices[i]:前一天结束时没有股票那么第i天需要买入)
//     base case: dp[0][0]=0，dp[0][1]=−prices[0]
int maxProfitII(vector<int>& prices) {
    int n = prices.size();
    int dp[n][2];
    dp[0][0] = 0, dp[0][1] = -prices[0];
    for (int i = 1; i < n; ++i) {
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
    }
    return dp[n - 1][0]; // 最后一天不持股的收益就是最大收益
}

int maxProfitII_v(vector<int>& prices) { // 状态压缩
    int n = prices.size();
    int dp0 = 0, dp1 = -prices[0];
    for (int i = 1; i < n; ++i) {
        int newDp0 = max(dp0, dp1 + prices[i]);
        int newDp1 = max(dp1, dp0 - prices[i]);
        dp0 = newDp0;
        dp1 = newDp1;
    }
    return dp0;
}

// 3. 买卖股票的最佳时机III 最多可进行2次交易
// 题目描述：设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易（不能同时参与多笔交易）
//
// III只能完成两笔交易：思路：那么每天完成后 将有以下5个可能的状态
//  (1)未进行过任何操作；
//  (2)只进行过一次买操作；
//  (3)进行了一次买操作和一次卖操作，即完成了一笔交易；
//  (4)在完成了一笔交易的前提下，进行了第二次买操作；
//  (5)完成了全部两笔交易。
int maxProfitIII(vector<int>& prices) {
    int n = prices.size();
    // 初始状态
    int buy1 = -prices[0], sell1 = 0;
    int buy2 = -prices[0], sell2 = 0;
    // 状态转移
    for (int i = 1; i < n; ++i) {
        buy1 = max(buy1, -prices[i]); // 假设当前买入
        sell1 = max(sell1, buy1 + prices[i]); // 假设当前卖出
        buy2 = max(buy2, sell1 - prices[i]);
        sell2 = max(sell2, buy2 + prices[i]);
    }
    return sell2; // 最后最大利益一定是进行两次交易
} // 时间复杂度O(N),空间O(1)



// 4. 买卖股票的最佳时机VI, 限制最多完成k笔交易.
// 思路：与其余的股票问题类似，我们使用一系列变量存储「买入」的状态，再用一系列变量存储「卖出」的状态，通过动态规划的方法即可解决本题。
//限制条件
//  先买入才能卖出
//  不能同时参加多笔交易，再次买入时，需要先卖出
//  k >= 0才能进行交易，否则没有交易次数
//定义操作
//  买入
//  卖出
//  不操作
//定义状态
//i: 天数
//k: 交易次数，每次交易包含买入和卖出，这里我们只在买入的时候需要将 k - 1
//0: 不持有股票
//1: 持有股票
//  dp[i][k][0]//第i天 还可以交易k次 手中没有股票
//  dp[i][k][1]//第i天 还可以交易k次 手中有股票
//最终的最大收益是dp[n - 1][k][0]而不是dp[n - 1][k][1]，因为最后一天卖出肯定比持有收益更高
int maxProfitVI(vector<int>& prices, int k) {
    int n = prices.size();
    if(n == 0) return 0;
    // 第i天，还可以交易k次，手中持股/手中不持股
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(k+1, vector<int>(2,-1e9)));
    // 定义初始情况
    dp[0][0][0] = 0;
    dp[0][0][1] = -prices[0];
    for(int i=1; i < n; i++){
        //小于等于天数或者小于等于k
        for(int j=1; j<=i && j<=k; j++){
            // 啥都不干
            dp[i][0][0] = 0;
            // 当前手上持股情况，操作的次数要小于天数，前面已经做了统一处理，这里的开始是从0开始所以要-1
            dp[i][j-1][1] = max(dp[i-1][j-1][0] - prices[i], dp[i-1][j-1][1]);
            // 当前手上不持股，操作次数可以等于天数，当天买当天卖
            dp[i][j][0] = max(dp[i-1][j-1][1] + prices[i], dp[i-1][j][0]);
        }
    }
    // 输出结果
    int res=0;
    for(int i=0;i<=k;i++){
        res=max(res, dp[n-1][i][0]);
    }
    //结果
    return res;
}// 时间复杂度：O(nk)
// （竞赛圈常用的）"wqs二分法"可以优化到 O(n*logC), C是数组prices 中的最大值。对凸包斜率进行二分


// 5. 买卖股票的最佳时机含冷冻期
// 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
//   卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if(n == 0) return 0;
    // dp[i][0]: 手上持有股票的最大收益
    // dp[i][1]: 手上不持有股票，并且处于冷冻期中的累计最大收益
    // dp[i][2]: 手上不持有股票，并且不在冷冻期中的累计最大收益
    vector<vector<int>> dp(n, vector<int>(3,0));
    dp[0][0] = -prices[0]; // 初始化
    for (int i = 1; i < n; ++i) {
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i]);
        dp[i][1] = dp[i - 1][0] + prices[i]; // 在冷冻期无法卖出
        dp[i][2] = max(dp[i - 1][1], dp[i - 1][2]); // 不在冷冻期
    }

    return max(dp[n - 1][1], dp[n - 1][2]);
}


// 6. 买卖股票的最佳时机含手续费
//  这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
//  思路参考II    定义定义状态dp[i][0] 表示第 i 天交易完后手里没有股票（也就是第i天卖出）的最大利润，
//          dp[i][1] 表示第 i 天交易完后手里持有一支股票（也就是第i天不卖）的最大利润（i 从 0 开始）。
int maxProfit_with_exchange_fee(vector<int>& prices, int fee) {
    int n = prices.size();
    int dp[n][2];
    dp[0][0] = 0, dp[0][1] = -prices[0];
    for (int i = 1; i < n; ++i) {
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
    }
    return dp[n - 1][0]; // 最后一天不持股的收益就是最大收益
}





#endif //DATASTRUCT_ALGORITHM_DP_BUY_STOCKS_H

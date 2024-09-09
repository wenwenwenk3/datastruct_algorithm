//
// Created by kai.chen on 2021/12/8.
//
// 一维dp.
//      1.LIS 最长递增子序列的长度
//      2.信封嵌套
//      3.最大子数组和
//          // 升级变体3.1：二叉树的最大路径和
//          // 变体3.2： 最大子数组乘积
//          // 变体3.3： 环形子数组最大和
//          // 变体3.4： 分割数组以得到最大和
//      4.字符可能的解码方法总数 A-1 Z-26
//      5.数字的可能翻译方法  0-a 25-z
//
//      6.爬楼梯
//      7.买卖股票的最佳时机、买卖股票的最佳时机 II
//      8.摊烧饼
//      9.和为k的子数组 (见array_op.h) 形成3的最大倍数 backtrack.h
//
//      10.接雨水 （解法见 monstack单调栈）
//      11.滑动窗口最大值 （解法见 monqueue单调队列）
//
//
//      12. 最短无序连续子数组
//      13. 跳跃游戏 到家的最少跳跃次数 jump_game.h
#ifndef DATASTRUCT_ALGORITHM_DP_1D_H
#define DATASTRUCT_ALGORITHM_DP_1D_H
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <numeric>
using namespace std;

// 1. LIS 最长递增子序列的长度
// (定义dp[i]表示以nums[i]为结尾的最长递增子序列的长度)
int length_of_LIS(vector<int>& nums){
    vector<int> dp(nums.size(), 1);
    for(int i = 0; i < nums.size(); i++){
        for(int j = 0; j < i; j++){
            if(nums[j] < nums[i]){
                dp[i] = max(dp[i], dp[j]+1);
            }
        }
    }
    int res = 0;
    for(auto item: dp){
        res = max(res, item);
    }
    return res;
} // 时间复杂度：O(n^2)
// 空间：O(n)

// 2.信封嵌套
// (每次合法的嵌套是大的套小的，相当于最长递增的子序列的长度)
bool compare(vector<int> a, vector<int> b){
    return a[0] == b[0]?a[1]>b[1]:a[0]<b[0];
}
int max_envelopes(vector<vector<int>>& envelopes){

    sort(envelopes.begin(), envelopes.end(), compare);
    //         sort(envelopes.begin(), envelopes.end(), [](const auto& a, const auto& b) {
    //            return a[0] < b[0] || (a[0] == b[0] && a[1] > b[1]);
    //        });

    vector<int> dp(envelopes.size(),1);
    for(int i=0; i < envelopes.size(); i++){
        for(int j=0; j < i; j++){
            if(envelopes[i][1] > envelopes[j][1]){
                dp[i] = max(dp[i], dp[j]+1);
            }
        }
    }

    int maxV = 1;
    for(int it=0; it< dp.size(); ++it){
        maxV=max(maxV, dp[it]);
    }
    return maxV;
} // 时间复杂度：O(N^2), 空间复杂度：O(N)

// 基于二分的dp
int maxEnvelopes(vector<vector<int>>& envelopes) {
    int n = envelopes.size();
    sort(envelopes.begin(), envelopes.end(), [](const vector<int>& a, const vector<int>& b) {
        return a[0] == b[0]?a[1]>b[1]:a[0]<b[0];
    });

    vector<int> dp = {envelopes[0][1]};
    for (int i = 1; i < n; ++i) {
        int num = envelopes[i][1];
        if (num > dp.back()) {
            dp.push_back(num);
        }
        else {
            // lower_bound二分查找的lower版本，时间复杂度logN
            // 返回[l, r)有与num相等元素的位置，否则返回该插入的位置右侧
            auto it = lower_bound(dp.begin(), dp.end(), num);
            *it = num;
        }
    }
    return dp.size();
}// 时间复杂度：O(N * logN), 空间复杂度：O(N)
void test_maxEnvelopes(){
    vector<vector<int>> env = {{5,4},{6,5},{6,7},{2,3}};
    cout<<maxEnvelopes(env)<<endl;
}


// 3.最大子数组和
int max_sub_array(vector<int>& nums){
    if(nums.empty()) return 0;
    vector<int> dp(nums.size());

    dp[0] = nums[0];
    for(int i = 1; i < nums.size(); i++){
        dp[i] = max(nums[i], dp[i-1]+nums[i]);
    }

    int max = dp[0];
    for(auto item:dp){
        if(item>max) max = item;
    }
    return max;
}

int maxSubArray_v2(vector<int>& nums) { // 状态压缩优化
    if(nums.empty()) return 0;
    int dp_0 = nums[0], dp_1 = nums[0], res=dp_1;
    for(int i=1; i<nums.size(); i++){
        dp_1 = max(nums[i], dp_0+nums[i]);
        res = max(res, dp_1);
        dp_0 = dp_1;
    }
    return res;
}
int maxSubArray_v3(vector<int>& nums) { // 更优雅的写法
    // 调用方保证 nums 不为空
    int n = nums.size();
    int sum = 0; // 当前的为结尾的最大子串和
    int ans = INT_MIN; // 当前的全局最大子串和
    for(int i = 0; i < n; ++i){
        sum = nums[i] + max(sum, 0);
        ans = max(ans, sum);
    }
    return ans;
}

// 升级变体3.1：二叉树的最大路径和
//  题目描述：路径和 是路径中各节点值的总和。该路径 至少包含一个 节点，且不一定经过根节点。
//  思路：和求最大子数组和一样，递归的求左右子节点
//    注意：root = [1,2,3] 最大路径和是 2 + 1 + 3 = 6
int maxSum = INT_MIN;
int backtrack(TreeNode* root) {
    if (root == nullptr) {
        return 0;
    }
    // 递归计算左右子节点的最大贡献值
    // 做选择，取左子节点还是不取，取右节点还是不取
    // 只有在最大贡献值大于 0 时，才会选取对应子节点
    int leftSum = max(backtrack(root->left), 0);
    int rightSum = max(backtrack(root->right), 0);

    // 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
    int priceNewpath = root->val + leftSum + rightSum;
    // 更新答案
    maxSum = max(maxSum, priceNewpath);
    // 返回节点的最大贡献值
    return root->val + max(leftSum, rightSum);
}
int maxPathSum(TreeNode* root) {
    backtrack(root);
    return maxSum;
} //时间复杂度：O(N)，其中 N 是二叉树中的节点个数。对每个节点访问不超过 2 次。
// 空间复杂度：O(N)，其中 NN 是二叉树中的节点个数。空间复杂度主要取决于递归调用层数，最大层数等于二叉树的高度，最坏情况下，二叉树的高度等于二叉树中的节点个数。

// 3.1.1 监控二叉树所需的最小摄像头数量

class SolutionJiankong {
private:
    int result;
    int traversal(TreeNode* cur) {
        // 空节点，该节点有覆盖
        if (cur == NULL) return 2;

        int left = traversal(cur->left);    // 左
        int right = traversal(cur->right);  // 右

        // 情况1
        // 左右节点都有覆盖
        if (left == 2 && right == 2) return 0;

        // 情况2
        // left == 0 && right == 0 左右节点无覆盖
        // left == 1 && right == 0 左节点有摄像头，右节点无覆盖
        // left == 0 && right == 1 左节点有无覆盖，右节点摄像头
        // left == 0 && right == 2 左节点无覆盖，右节点覆盖
        // left == 2 && right == 0 左节点覆盖，右节点无覆盖
        if (left == 0 || right == 0) {
            result++;
            return 1;
        }

        // 情况3
        // left == 1 && right == 2 左节点有摄像头，右节点有覆盖
        // left == 2 && right == 1 左节点有覆盖，右节点有摄像头
        // left == 1 && right == 1 左右节点都有摄像头
        // 其他情况前段代码均已覆盖
        if (left == 1 || right == 1) return 2;

        // 以上代码我没有使用else，主要是为了把各个分支条件展现出来，这样代码有助于读者理解
        // 这个 return -1 逻辑不会走到这里。
        return -1;
    }

public:
    int minCameraCover(TreeNode* root) {
        result = 0;
        // 情况4
        if (traversal(root) == 0) { // root 无覆盖
            result++;
        }
        return result;
    }
};


// 变体3.2 最大子数组乘积
// 看起来和最大子数组很像，但不能那样理解。这里主要难点在于有正负号的处理，
// 思路：维护当前的最小和最大值乘积
int maxProduct(vector<int>& nums) {
    vector <int> dpMax(nums);
    vector <int> dpMin(nums);
    for (int i = 1; i < nums.size(); ++i) {
        // 要么等于dp[i-1] * nums[i] 这个时候nums[i]等于正数, 要么等于max(nums[i]自成一派或者 找i-1为结尾最小值相乘)
        dpMax[i] = max(dpMax[i - 1] * nums[i], max(nums[i], dpMin[i - 1] * nums[i]));
        dpMin[i] = min(dpMin[i - 1] * nums[i], min(nums[i], dpMax[i - 1] * nums[i]));
    }
    int max = dpMax[0];
    for(auto item:dpMax){
        if(item>max) max = item;
    }
    return max;
}
int maxProduct_v2(vector<int>& nums) { // 状态压缩优化
    int maxProd = nums[0], minProd = nums[0], res = nums[0];
    for (int i = 1; i < nums.size(); ++i) {
        int curMax = maxProd, curMin = minProd;
        maxProd = max(curMax * nums[i], max(nums[i], curMin * nums[i]));
        minProd = min(curMin * nums[i], min(nums[i], curMax * nums[i]));
        res = max(maxProd, res); //
    }
    return res;
}

// 3.3 环形子数组的最大和
// Kadane 算法：在数组或滑动窗口中找到子串和的最大值或最小值的 O(N) 算法，它基于动态规划
//
//        dp[i] := [0..i] 中，以 nums[i] 结尾的最大子串和
//        状态转移：dp[i] = nums[i] + max(dp[i - 1], 0)
//        base case：dp[0] = nums[0]
// 当数组可以循环时，子段结果可能是"单区间"或"双区间", Kadane算法结果是单区间的。
//  对于双区间[0, j], [i, n - 1]的情况，两个区间表示的子段和取得最大值时，意味着子段 [j + 1, i - 1] 的和取到最小值。（毕竟sum(A) 是固定的）

// 更一般化的思路：前缀和 + 单调队列 【能推广到很多类似问题】（如array_string的和为k的子数组，至少为k的最短子数组)
//      先计算 A[0..n-1] 的前缀和 sums[0..n] ，
//      枚举每个前缀和的下标j = [1,n]，现在要找 j-i <= N 时最大的 sums[j] - sums[i]，即找最小的 sums[i]。
//      要想O(1) 地找到每个 j 的左侧最值及其下标 i，其中 i 的范围需要满足某些条件(这里是 j - i <= N)。这是典型的单调队列的场景：即滑动窗口的最值问题
// 具体解法见 算法进阶：monqueue.h
int maxSubarraySumCircular(vector<int>& nums) {
    int n = nums.size();
    if(n == 1) return nums[0];
    int total_sum = nums[0];
    // 单区间最大子段和
    int sum1 = nums[0], res_single_seg = nums[0];
    for(int i = 1; i < nums.size(); i++){
        total_sum += nums[i];
        sum1 = nums[i] + max(sum1, 0);
        res_single_seg = max(res_single_seg, sum1);
    }

    if(n == 2) return res_single_seg;
    // 双区间的最大字段和 = total_sum - res_min_single_seg
    int sum2 = nums[1], res_min_single_seg = nums[1];
    for(int i = 2; i < n - 1; ++i){
        sum2 = nums[i] + min(sum2, 0);
        res_min_single_seg = min(res_min_single_seg, sum2);
    }
    int res_double_seg = total_sum - res_min_single_seg;

    return max(res_single_seg, res_double_seg);
}

void testmaxSubarraySumCircular(){
    int a[] = {1,-2,3,-2};
    vector<int> nums(a, a+sizeof(a)/sizeof(a[0]));
    cout<<"max sum:"<<maxSubarraySumCircular(nums)<<endl;
}

// 变体3.4： 分割数组以得到最大和
// 题目描述：给你一个整数数组 arr，请你将该数组分隔为长度最多为 k 的一些（连续）子数组。分隔完成后，每个子数组的中的所有值都会变为该子数组中的最大值。
// 思路： dp[n] 表示前n个元素分割为长度小于k的连续子数组的最大和
//    dp[n]=max(dp[n-1]+max(arr[n])*1, dp[n-2]+max(arr[n], arr[n-1])*2, ... , dp[n-k]+max(arr[n], arr[n-k+1])*k);
//    即：for i in range(1,k):
//          dp[n] = max(dp[n], dp[n-i]+max(arr[n], arr[n-i+1])*i);
//    base case: n < k时，n * maxInrange()
long long maxres = 0;
int maxInRange(vector<int>& arr, int start, int end) { // [start, end]
    int maxres = 0;
    for (int i = start; i <= end; i++) {
        maxres = max(maxres, arr[i]);
    }
    return maxres;
}
int maxSumAfterPartitioning(vector<int>& arr, int k) {
    // 输入保证数组不为空
    int n = arr.size();
    vector<int> dp(n);
    for (int i = 0; i < k; i++) {
        dp[i] = maxInRange(arr, 0, i) * (i+1);
    }
    for (int i = k; i < n; i++) {
        for (int j = 1; j <= k; j++) {
            dp[i] = max(dp[i - j] + maxInRange(arr, i - j + 1, i) * j, dp[i]);
        }
    }
    return dp[n - 1];
} // 时间复杂度： O( N*k* k), 空间复杂度: O (n)

// 变体3.5: 分割数组让子数组各自和的最大值最小
// 题目描述：
//      输入：nums = [7,2,5,10,8], m = 2
//      输出：18
//      说明：一共有四种方法将 nums 分割为 2 个子数组。其中最好的方式是将其分为 [7,2,5] 和 [10,8]
// 思路：令 f[i][j] 表示将数组的前 i 个数分割为 j 段所能得到的最大连续子数组和的最小值
// https://leetcode-cn.com/problems/split-array-largest-sum/solution/fen-ge-shu-zu-de-zui-da-zhi-by-leetcode-solution/
int splitArray(vector<int>& nums, int m) {
    int n = nums.size();
    vector<vector<long long>> f(n + 1, vector<long long>(m + 1, LLONG_MAX));
    vector<long long> sub(n + 1, 0);
    for (int i = 0; i < n; i++) {
        sub[i + 1] = sub[i] + nums[i];
    }
    f[0][0] = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= min(i, m); j++) {
            for (int k = 0; k < i; k++) {
                f[i][j] = min(f[i][j], max(f[k][j - 1], sub[i] - sub[k]));
            }
        }
    }
    return (int)f[n][m];
} // 时间复杂度： O(n*n*m), 空间复杂度：O(n*m)

// 思路2：二分+贪心
// 二分的上界为数组 nums 中所有元素的和, 下界为数组 nums 中所有元素的最大值
int split(vector<int>& nums, int temMax){ // 函数split在定义域内（temMax的取值范围内）单调递减
    int res = 1; // 最少也是所有元素一组
    int temSum = 0;
    for(int x:nums){
        temSum += x;
        if(temSum > temMax){
            res++;
            temSum = x;
        }
    }
    return res;
}
int splitArray2(vector<int>& nums, int m) {
    int l = *max_element(nums.begin(), nums.end()); // 子数组的最小和为最大的元素，注意该函数返回的是迭代器
    int r = accumulate(nums.begin(), nums.end(), 0); // 子数组的最大和为所有元素的和，此处为右闭写法

    while(l < r){ // 找左边界
        int mid = l + (r - l) / 2; // 防溢出
        if(split(nums, mid) <= m){ // 注意，是小于等于都满足条件
            r = mid;
        }else{ // 对于定义域，只有两个区块，前一半不满足，后一半满足，所以就俩情况。
            l = mid + 1; // 注意，由于是单调递减，所以在split>m时，反而向右移动l而不是向左移动r
        }
    }
    return l;
} // 时间复杂度: O(n×log(sum−maxn)) 当数据比较集中的时候更高效
// 空间：O(1)



// 4.字符解码方法
//题目描述：
// 'A' -> 1 、'B' -> 2、 ..、'Z' -> 26
// 请计算字符串 s  解码 方法的 总数
// 思路：
//  当使用s[i]作为单独字符解码时：只要s[i] != 0
//   dp[i] = dp[i-1]
//  当使用s[i]和前一个字符解码时：只要s[i-2] - '0') * 10 + (s[i-1] - '0') <= 26
//   dp[i] = dp[i-1] + dp[i-2]
int numDecodings(string s) {
    int len = s.length();
    vector<int> dp(len + 1, 0); // 注意f[x]表示前x个字符(最终答案是f[n])的解码方法数，f[0]=1
    dp[0] = 1;
    for(int i = 1; i <= len; i++){
        if(s[i-1] != '0'){
            dp[i] += dp[i - 1];
        }
        if(i>1 && s[i-2] != '0'){
            if((s[i-2] - '0') * 10 + (s[i-1] - '0') <= 26){
                dp[i] += dp[i - 2];
            }
        }
    }

    return dp[len];
} // 时间复杂度O(n), 空间复杂度O(n)

int numDecodings_v2(string s) { // 状态压缩优化（滚动数组）
    int len = s.length();
    int fp = 0, f0 = 1, fx = 0; // f[i-2],f[i-1],f[i]

    for(int i = 1; i <= len; i++){
        fx = 0;
        if(s[i-1] != '0'){
            fx += f0;
        }
        if(i>1 && s[i-2] != '0' && ((s[i-2] - '0')*10 + (s[i-1] - '0') <= 26)){
            fx += fp;
        }
        fp = f0;
        f0 = fx;
    }
    return fx;
} // 时间复杂度O(n), 空间复杂度O(1)

void test_numDecodings(){
    int res = numDecodings("506");
    int res2 = numDecodings_v2("06");
    cout<<"num of decode methods:"<<res<<","<<res2<<endl;
}

// 5.数字的可能的翻译方法总数  0-a 25-z
// 思路：这道题和上一道相似，存在重叠子问题，但明显从后往前递归会更高效
//     12122 --> dpTable: [8,5,3,2,1]
//                    3 : "1,2,2", "12,2", "1,22"
//                  5 : "2,1,2,2", "2,12,2", "2,1,22", "21,2,2", "21,22"
//   dp[i] = dp[i+1]+dp[i+2]
//      (小提示，这很像求斐波那契数列只不过加了个判断，而且可以状态压缩)
int getNumsTranslate(string& num){
    int n = num.size();
    vector<int> res(n,1);
    // base case: res[n-1] = 1

    for(int i = n-2; i >= 0; i--){
        int x = (num[i] - '0')*10 + (num[i+1] - '0');
        if(x>=10 && x<= 25){
            res[i] = i==n-2 ? res[i+1] + 1: res[i+1] + res[i+2];
        }else{
            res[i] = res[i+1];
        }
    }
    return res[0];
}
int numsTranslate(int num){
    if(num < 0) return 0;
    string numstr = to_string(num);
    return getNumsTranslate(numstr);
}
void testnumsTranslate(){
    cout<<numsTranslate(624)<<endl;
}

// 6.爬楼梯 / 青蛙跳台阶
// 每次你可以爬 1 或 2 个台阶，有多少种不同的方法可以爬到楼顶n
//      f(i) = f(i-1)+f(i-2)
//  base case: f0 = 1, f1 = 1, f2 = 2
int climbStairs(int n) {
    int f0 = 0, f1 = 0;
    long long fi = 1;
    for (int i = 1; i <= n; ++i) {
        f0 = f1;
        f1 = fi;
        fi = (f0 + f1) % (1000000007);
    }
    return fi;
}
// 延伸爬楼梯，假设每次可以爬x步(x in steps[])，有多少种方法爬到楼顶
// dp[i]表示爬到第i层楼梯的方法数，dp[i] = cigma(dp[i-x])  (x in steps)
int climbStairsII(vector<int>& steps, int target){
    vector<int> dp(target+1, 0);
    dp[0] = 1;
    for(int i = 0; i <= target; i++){
        for(const auto & step : steps){
            if(i - step >= 0) dp[i] += dp[i-step];
        }
    }
    return dp[target];
}


// 8.摊烧饼
// 题目描述：n块面积大小不一的烧饼cakes[]
// 思路：题目明显有递归性质，找到前n个烧饼的最大的那个，翻转到最底下，递归调用pancakeSort(n-1)
//      其中将最大烧饼翻转到最底下过程可以这样做，先反转到最上面，再整体翻转将最上面的一个（也就是最大的烧饼）翻转到最下面
//  base case: n=1, 不需要翻转
void reverse(int a[], int i, int j){
    while(i<j){
        swap(a[i], a[j]);
        i++,j--;
    }
}
vector<int> resOrder; // 记录每次翻转操作动作序列
void sortcake(int cakes[], int n){
    if( n== 1) return;
    int maxCakeIndex = 0; // 寻找最大烧饼的下标
    int maxCake = 0;
    for(int i = 0; i <n ; i++){
        if(cakes[i] > maxCake) {
            maxCake = cakes[i];
            maxCakeIndex = i;
        }
    }
    // 先反转到最上面
    reverse(cakes, 0, maxCakeIndex);
    resOrder.push_back(maxCakeIndex+1); // 记录，下标加一表示翻转的个数
    // 将最上面的一个（也就是最大的烧饼）翻转到最下面
    reverse(cakes, 0, n-1);
    resOrder.push_back(n);
    sortcake(cakes, n-1);
}
vector<int>& pancakeSort(vector<int>& cakes){
    sort(cakes.begin(), cakes.end());
    return resOrder;
} // 时间复杂度：O(N^2) 递归次数n,每次都是for循环n次



// 9. 和为k的子数组
// 题目描述：求nums中一共有几个和为k的子数组
// 思路：见array_string_op  3. 和为k的子数组
int subArraySum(vector<int>& nums, int k){
    return 0;
}


// 10.11 见monstack monqueue


// 12. 需要排序的最短子数组
// 原问题是：求出要去除的最小子数组长度让原无序数组的前后[0,x],[y,N-1]两段区间保持有序，换一种理解就无非是求出数组中需要排序的最短子数组
//
// 思路：
//      把这个数组分成三段, 左段和右段是标准的升序数组，中段数组虽是无序的，但满足最小值大于左段的最大值，最大值小于右段的最小值。
//      那么问题就是 找中段的左右边界，我们分别定义为begin 和 end
//    第一步：从左到右维护一个最大值max,在进入右段之前，那么遍历到的nums[i]都是小于max的，我们求的end就是遍历中最后一个小于max元素的位置；
//    第二步：同理，从右到左维护一个最小值min，在进入左段之前，那么遍历到的nums[i]也都是大于min的，要求的begin也就是最后一个大于min元素的位置。
int findUnsortedSubarray(vector<int>& nums) { // [2,｜6,4,8,10,9｜,15]
    int n = nums.size();
    if(n <= 1) return 0;
    // 找右边第一个出现逆序的位置rightIdx
    int min = nums[n-1], begin = -1;
    for(int i = n-2; i >= 0; i--){
        if(nums[i] > min) { // 不断找到最后一个大于min元素的位置
            begin = i;
        }else{ // 否则更新min
            min = nums[i];
        }
    }
    if(begin == -1){ // 右指针跑完都没有大于min元素的元素，说明数组有序
        return 0;
    }

    // 继续找左边第一个出现逆序的位置leftIdx
    int max = nums[0], end = -1;
    for (int i = 1; i < n; i++) {
        if(nums[i] < max){ // 不断找到最后一个小于max元素的位置
            end = i;
        }else{
            max = nums[i];
        }
    }

    return end - begin + 1; // 输出它的长度
}
void testfindUnsortedSubarray(){
    int a[] = {2,6,4,8,10,9,15};
    vector<int> nums(a, a+sizeof(a)/sizeof(a[0]));
    cout<<"unsorted len: "<<findUnsortedSubarray(nums)<<endl;
}


// 13. 跳跃游戏
// 题目描述：给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
//      数组中的每个元素代表你在该位置可以跳跃的最大长度。请判断是否能够到达最后一个下标。
//      例如：输入：nums = [2,3,1,1,4]  输出：true   解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
// 思路：
//      贪心：从x能否到y，只需要判断 x + nums[x] ≥ y，那么位置y可以到达。
//          只需要依次扫描数组中的每一个位置，并实时维护一个 最远可以到达的位置maxStep，当它大于n-1即可
bool canJump(vector<int>& nums) {
    int n = nums.size();
    int maxStep = 0;
    for (int i = 0; i < n; ++i) {
        // 只有当前位置小于等于maxStep，当前位置才是可达。否则根本不可达
        if (i <= maxStep) {
            maxStep = max(maxStep, i + nums[i]);
            if (maxStep >= n - 1) {
                return true;
            }
        }
    }
    return false;
}

// 13.1  跳跃游戏II
// 这次的目标是使用最少的跳跃次数到达数组的最后一个位置。
//  （假设你总是可以到达数组的最后一个位置。）
// 输入: nums = [2,3,1,1,4]  输出: 2  解释: 跳到最后一个位置的最小跳跃数是 2。



#endif //DATASTRUCT_ALGORITHM_DP_1D_H

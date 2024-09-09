//
// Created by kai.chen on 2022/1/11.
//
//       0. 丑数、丑数II
//       1. 约瑟夫环、
//       2. 汉诺塔、
//       3. 加油站、
//       4. 空瓶换酒
//
//       5. 灯泡开关

//       6. x的平方根 sqrt(x)
//       7. x的幂运算 pow(x, n)
//
//       8. 等差数列划分问题（子数组版 & 子序列版）
//
//       9. 素数判断
//
//       10. 完全平方数
//       11. 按权重生成随机数
//       12. 圆内均匀生成点
//       13. 非重叠矩形中随机生成点
//
//       14. 分发糖果
//       15. 计算数字序列中某一位的数字
//
//       .糖果交换 见 two_sum.h
//       .字符串加法，36进制加法，rand7生成rand10 （见two_sum.h）
//       .Nim游戏，&进阶：100 game 能赢游戏 （见array_subset.h）
//       .石子游戏，&进阶：预测赢家 （见array_subset.h）
//       .24点游戏
//
#ifndef DATASTRUCT_ALGORITHM_MATH_PROBLEM_H
#define DATASTRUCT_ALGORITHM_MATH_PROBLEM_H
#include <unordered_set>
#include <queue>
using namespace std;

// 0. 丑数
//  题目描述：丑数 就是只包含质因数 2、3 和/或 5 的正整数。
// 朴素解法：除以2除以3除以5判断最后是不是1。时间复杂度：当 n 是以 2 为底的对数时，需要除以 logn 次。复杂度为 O(logn)
bool isUgly(int n) {
    if (n <= 0) { // 如果 n 不是正整数（即小于等于 0）：必然不是丑数，直接返回 false。
        return false;
    }
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    return n == 1;
}

// 0.1 第 n 个丑数（丑数II）
// 思路：优先级队列
//      - 起始先将最小丑数 1 放入队列
//      - 每次从队列取出最小值 x，然后将 x 所对应的丑数 2x、3x 和 5x 进行入队。
//      - 不断的循环多次，第 n 次出队的值即是答案。
//    为了防止同一丑数多次进队，我们需要使用数据结构 Set 来记录入过队列的丑数
int nthUglyNumber(int n) {
    int nums[] = {2, 3, 5};
    unordered_set<long> s;
    priority_queue<long, vector<long>, greater<long>> pq;
    s.insert(1);
    pq.push(1);
    for (int i = 1; i <= n; i++){
        long x = pq.top();
        pq.pop();
        if (i == n) return (int)x;
        for (int num : nums){
            long t = num * x;
            if (!s.count(t)){
                s.insert(t);
                pq.push(t);
            }
        }
    }
    return -1;
} // 时间复杂度O(n*logn), 每次往priority_queue添加元素是O(logn)，
//  空间复杂度O(N)

// O(n)时间的进阶解法：用三个指针
//  思路： 从解法一中不难发现，我们「往后产生的丑数」都是基于「已有丑数」而来（使用「已有丑数」乘上「质因数」2、3、5）。
//      任何序列可以看作 s2,s3,s5 三个有序序列归并而来：
int nthUglyNumber(int n) {
    // ans 用作存储已有丑数（从下标 1 开始存储，第一个丑数为 1）
    vector<int> res(n + 1);
    res[1] = 1;
    // 由于三个有序序列都是由「已有丑数」*「质因数」而来
    // i2、i3 和 i5 分别代表三个有序序列当前使用到哪一位「已有丑数」下标（起始都指向 1）
    for (int i2 = 1, i3 = 1, i5 = 1, idx = 2; idx <= n; idx++) {
        // res[sX] * X
        int a = res[i2] * 2, b = res[i3] * 3, c = res[i5] * 5;
        // 将三个有序序列中的最小一位存入「已有丑数」序列，并将其下标后移
        int minValue = min(a, min(b, c));
        // 由于可能不同有序序列之间产生相同丑数，因此只要一样的丑数就跳过（不能使用 else if ）
        if (minValue == a) i2++;
        if (minValue == b) i3++;
        if (minValue == c) i5++;
        res[idx] = minValue;
    }
    return res[n];
}

// 0.3 丑数 III
//   题目描述：给你四个整数：n 、a 、b 、c ，请你设计一个算法来找出第 n 个丑数。
//      注意这次：丑数是可以被 a 或 b 或 c 整除的 正整数 。
typedef long long ll;
// greatest common divisor 最大公约数
ll gcd(ll a, ll b) {
    return b == 0 ? a : gcd(b, a % b);
}
// least common multiple 最小公倍数
ll lcm(ll a, ll b) {
    return a * b / gcd(a, b);
}

int nthUglyNumber(int n, int a, int b, int c) {
    ll ab = lcm(a, b), ac = lcm(a, c), bc = lcm(b, c);
    ll abc = lcm(ab, c);
    //本题结果在 [1, 2 * 10^9] 的范围内，优化为 [min({a,b,c}), 2 * 10^9]
    ll l = min(a,min(b,c))-1, r = 2e9+1;
    while (l < r) {
        ll mid = (l+r)>>1;
        // 计算 cnt 为[1,mid]中的丑数个数
        // 容斥原理：a并b并c = a + b + c - a交b - b交c - a交c + a交b交c
        ll cnt = mid / a + mid / b + mid / c - mid / ab - mid / ac - mid / bc + mid / abc;
        if (cnt < n) {
            l = mid;
        } else {
            r = mid;
        }
    }
    return (int)r;
}

// 最大公因数等于 K 的子数组数目
int subarrayGCD(vector<int>& nums, int k) {
    int n = nums.size();
    int res = 0;
    for(int i = 0; i < n; i++){
        int tempgcd = nums[i];
        for(int j = i; j < n; j++){
            tempgcd = __gcd(tempgcd, nums[j]);
            if(tempgcd == k) res++;
        }

    }
    return res;
}
// 最小公倍数等于 K 的子数组数目
int subarrayLCM(vector<int>& nums, int k) {
    int n = nums.size();
    int ans = 0;
    for (int i = 0; i < n; i++) {
        long long templcm = nums[i];
        for (int j = i; j < n; j++) {
            long long g = gcd(l, nums[j]);
            templcm = templcm / g * nums[j];
            if (templcm == k) ans++;
                // 防止溢出
            else if (templcm > k) break;
        }
    }
    return ans;
}


// 1. 约瑟夫环
ListNode* JosephCircle(ListNode* pHead, int k){ // 1,2,3,4,5
    ListNode* cur = pHead, *tmp = nullptr;
    while(cur != cur->next){
        while(--k){
            cur = cur->next;
        }
        tmp = cur->next;
        cur->val = tmp->val;
        cur->next = tmp->next;
        free(tmp);
    }
    return cur;
}

int JosephCircleArrayVersion(int n, int m){
    int f = 0;
    for (int i = 2; i != n + 1; ++i){
        f = (m + f) % i;  //
    }
    return f;
}
// 递归版
int f(int n, int m) {
    if (n == 1) {
        return 0;
    }
    int x = f(n - 1, m);
    return (m + x) % n;
}


// 2. 汉诺塔
// n = 1 时，直接把盘子从 A 移到 C；
// n > 1 时，
//  先把上面 n - 1 个盘子从 A 移到 B（子问题，递归）；
//  再将最大的盘子从 A 移到 C；
//  再将 B 上 n - 1 个盘子从 B 移到 C（子问题，递归）。
void move(int n, vector<int>& A, vector<int>& B, vector<int>& C){
    if (n == 1){
        C.push_back(A.back());
        A.pop_back();
        return;
    }

    move(n-1, A, C, B);    // 将A上面n-1个通过C移到B
    C.push_back(A.back());  // 将A最后一个移到C
    A.pop_back();          // 这时，A空了
    move(n-1, B, A, C);     // 将B上面n-1个通过空的A移到C
}

void hanota(vector<int>& A, vector<int>& B, vector<int>& C) {
    int n = A.size();
    move(n, A, B, C);
}


// 3. 加油站：
//  题目描述：有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
//     从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i]
//  如果你可以绕环路行驶一周，则返回出发时加油站的编号（刚开始油箱为空），否则返回 -1
// 暴力解法O(n^2)，直接两次遍历
int canCompleteCircuit1(vector<int>& gas, vector<int>& cost) {
    for (int i = 0; i < cost.size(); i++) {
        int rest = gas[i] - cost[i]; // 记录剩余油量
        int index = (i + 1) % cost.size();
        while (rest > 0 && index != i) { // 模拟以i为起点行驶一圈
            rest += gas[index] - cost[index];
            index = (index + 1) % cost.size();
        }
        // 如果以i为起点跑一圈，剩余油量>=0，返回该起始位置
        if (rest >= 0 && index == i) return i;
    }
    return -1;
}

// 一次遍历的解法
// 情况一：如果gas的总和小于cost总和，那么无论从哪里出发，一定是跑不了一圈的
//
// 情况二：rest[i] = gas[i]-cost[i]为一天剩下的油，i从0开始计算累加到最后一站，如果累加没有出现负数，说明从0出发，油就没有断过，那么0就是起点。
//
// 情况三：如果累加的最小值是负数，汽车就要从非0节点出发，从后向前，看哪个节点能这个负数填平，能把这个负数填平的节点就是出发节点
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int curSum = 0;
    int min = INT_MAX; // 从起点出发，油箱里的油量最小值
    for (int i = 0; i < gas.size(); i++) {
        int rest = gas[i] - cost[i];
        curSum += rest;
        if (curSum < min) {
            min = curSum;
        }
    }
    if (curSum < 0) return -1;  // 情况1
    if (min >= 0) return 0;     // 情况2
    // 情况3
    for (int i = gas.size() - 1; i >= 0; i--) {
        int rest = gas[i] - cost[i];
        min += rest;
        if (min >= 0) {
            return i;
        }
    }
    return -1;
}

// 4. 空瓶换酒
// 题目描述：买了 n 瓶酒，假设每 m 个空瓶可以换一瓶酒，问最多能喝到多少瓶酒
//  方法一：模拟换酒过程
//  方法二：数学推理：
//      使用 m 个空酒瓶能够换得一瓶新酒（饮用数量加一，且新瓶子数量加一），即对于每换一瓶酒而言，会损失掉 m - 1 个瓶子
//       既然，每个回合损失的瓶子个数 m - 1 为定值，可以直接算出最大交换次数（额外饮用次数）cnt = n / (m  - 1)
int numWaterBottles(int n, int m) {
    int res = n; // 最起码能喝到 n 瓶

    while (n >= m) { // 当 n>= m时可以继续换酒
        int a = n / m, b = n % m; // 当前能兑换的酒数：n/m，本轮兑换后剩余的空瓶数 n%m
        res += a;
        n = a + b;
    }
    return res;
}
int numWaterBottles2(int n, int m) {
    int cnt = n / (m  - 1);
    return n % (m - 1) == 0 ? n + cnt - 1 : n + cnt;
}

// 5. 灯泡开关
// 整理一下题意：第 i 轮改变所有编号为 i 的倍数的灯泡的状态（其中灯泡编号从 1 开始）。
// 一个编号为 x 的灯泡经过 n 轮后处于打开状态的充要条件为「该灯泡被切换状态次数为奇数次」
//
// 约数个数为奇数，意味着某个约数在分解过程中出现了 2 次，即 k*k = x
// 问题最终转换为：在 [1,n] 中完全平方数的个数为多少。即sqrt(n)
int bulbSwitch(int n) {
    return (int)sqrt(n);
}


// 6. x的平方根
// 思路：二分法，无非就是求 k^2 <= x 时最大的k
int mySqrt(int x) {
    if(x == 0 || x == 1) return x;
    int left = 0, right = x;
    while(left <= right){ // 左闭右闭，等于要继续搜，因为[l,r]区间内还有一个元素。而退出条件是left=right+1
        int mid = left + (right-left) / 2;
        if((long long)mid * mid == x){
            return mid;
        }else if((long long)mid * mid < x){
            left = mid+1;
        }else {
            right = mid-1;
        }
    }
    return left-1; // 要求 如果平方根不是整数，输出只保留整数的部分，小数部分将被舍去。意味着该减1，当然return right也可以。
}

// 延伸：浮点数 求平方根
#define e 1e-6
double mySqrtDouble(double x){
    double left = 0, right = x;
    if(x < 1){ left = x, right = 1;}
    while(left <= right){
        double mid = l + (r-l)/2;
        if(abs(mid * mid - x) < e){
            return mid;
        }else if(mid * mid < x){
            left = mid+e;
        }else{
            right = mid-e;
        }
    }
    return right; // 其实不可能走到这，while不断循环一定能找到 mid*mid - x的 绝对值 小于e
}
// (牛顿迭代法: 求根号a的近似值，首先随便猜一个近似值x，然后不断令x等于x和a/x的平均数，迭代个六七次后x的值就已经相当精确了
//       缺点牛顿迭代存在一个初值选择的问题，选择得好会极大降低迭代的次数，选择得差效率也可能会低于二分法。)
// (Quake卡马克算法: “雷神之锤III”的电子游戏源码中Quake发明，更好的选择初值，让非常接近1/sqrt(n), )


// 7. x的幂运算
// 快速幂 + 递归
//      当我们要计算 x^n时，我们可以先递归地计算出 y = x^⌊n/2⌋ (下取整)
//      根据递归计算的结果，如果 n 为偶数，那么 x^n = y^2
//              如果 n 为奇数，那么 x^n = y^2 * x
//    递归的边界为 n = 0，任意数的 0 次方均为 1
double quickMul(double x, long long N) {
    if (N == 0) {
        return 1.0;
    }
    double y = quickMul(x, N / 2);
    return N % 2 == 0 ? y * y : y * y * x;
}

double myPow(double x, int n) {
    long long N = n;
    return N >= 0 ? quickMul(x, N) : 1.0 / quickMul(x, -N);
}


// 8. 等差数列划分问题 - 子数组
// 题目描述：如果一个数列至少有三个元素，并且任意两个元素之差相同，则称之为等差数列
//  求数组nums中所有为等差数组的 子数组 个数。
//  输入： [1,2,3,4]   输出：3   解释：[1,2,3],[2,3,4][1,2,3,4]
// 思路：
//   枚举 i 为等差数组左端点。找到所有 满足等差 的最右 右端点 j
//   假设[i, j]的长度为 len，  那么长度为 len 的等差子数组个数为 1，长度为len-1的为 2 ...
//      即等差子数组的总个数是：首项为1，末项为 len-3+1，公差为1的等差数列 求和 的结果
//     附：等差数列{an}的通项公式为：an=a1+(n-1)d， 前n项和公式为：Sn=n*a1+n(n-1)d/2或Sn=n(a1+an)/2 。”
int numberOfArithmeticSlices(vector<int>& nums){
    int n = nums.size();
    int res = 0;
    int i = 0;
    while(i < n-2){ // 枚举左端点 i
        int j = i + 2; // 右端点 j
        int d = nums[i+1] - nums[i]; // 公差
        if(nums[j] - nums[j-1] != d){
            i++;
            continue;
        }
        while(nums[j] - nums[j-1] == d){
            j++;
            if(j >= n) break;
        } // 这里 j 的退出条件的时候多加了一次，所以需要j--
        j--;
        int len = j - i + 1;
        // a1: 长度为len的子数组数量, an: 长度为3的子数组数量
        int a1 = 1, an = len - 3 + 1;
        int subarraycount = (a1+an) * an /2; // 首项为a1，末项为an，项数也为an

        res += subarraycount;
        i = j;
    }
    return res;
} // 时间复杂度O(n), 空间复杂度O(1)


// 8.1 等差数列划分问题 - 子序列
//  考虑 两个结尾 。这题困难题，简单分析即可。除非想为难你才希望你能做出来


// 8.2 最长定差子序列长度
// 思路：我们从左往右遍历 nums，并计算出以 nums[i] 为结尾的最长的等差子序列的长度，取所有长度的最大值，即为答案。
//  定义：dp[i] 表示以 nums[i] 为结尾的最长的等差子序列的长度
//      枚举 j (j<i) , 在i的左侧找到最近的 j 满足 nums[i] - nums[j] = d.  此时dp[i] = dp[j]+1
//    这样，相当于每次总是在最左侧找到 nums[j] = nums[i] - d 的元素，并取出 dp[j].
//  假设 nums[i] = x, 那么 重新定义 dp[x] 为以 x为结尾的最长等差子序列的长度。状态转移方程就是：
//      dp[x] = dp[x-d] + 1
int longestSubsequence(vector<int> &nums, int difference) {
    int d = difference;
    int ans = 0;
    unordered_map<int, int> dp;

    for (auto x: nums) {
        dp[x] = dp[x - d] + 1;
        ans = max(ans, dp[x]);
    }
    return ans;
}
//时间复杂度：O(n)，其中 n 是数组长度。
// 空间复杂度：O(n)。哈希表需要 O(n) 的空间


// 8.3 最长等差子序列的长度。
// 思路：基于 "8.2 最长定差子序列长度" 求解，枚举可能的 差 时间复杂度O(n^2)
int longestArithSeqLength(vector<int>& nums) {
    int n = nums.size();
    unordered_set<int> diffs;
    // 计算 所有可能的差
    for(int i = 0; i < n; i++){
        for(int j = i+1 ; j < n; j++){
            diffs.insert(nums[j]-nums[i]);
        }
    }

    int res = 0;
    for(int d : diffs){
        res = max(res, longestSubsequence(nums, d));
    }
    return res;
}


// 9. 素数判断
// 直接法 & 筛法
int Max = 5*1000000;
int countPrimes(int n) {
    int sum = 1; // 2是第一个素数
    for(int i = 3; i <= Max; i+=2){ // 所有的偶数已知不是素数
        int j;
        for(j = 2; j <= (int)sqrt(i); j++){
            if(i%j == 0) break;
        }
        sum++;
    }
    return sum;
}// 时间复杂度： O(n * n^(1/2)), 每个素数判断需要 根号n 复杂度，一共有n/2个数要判断
//

// 埃氏筛
// 如果x是素数，那么x的所有倍数都不是素数
int countPrimes(int n) {
    if(n < 2) return 0;
    int cnt = 0;
    int isPrime[n];
    memset(isPrime, 0,  sizeof(isPrime));

    for(int i = 2; i < n; i++){
        if(isPrime[i] == 0){// 筛完仍然没被标记为 1 的就是素数
            cnt++;
            for(int j = 2; j < n && i *j < n; j++){
                isPrime[i*j] = 1;
            }
        }
    }
    return cnt;
} // 时间复杂度：O(n * log log(n))，比较难解释。只要知道大于O(n)小于O(n^2)
// 空间复杂度：O(n), 需要一个n长的线性空间做筛条


// 线性筛-竞赛版本(理论较难解释，非面试范畴)
int countPrimes3(int n) {
    if (n < 2) {
        return 0;
    }
    int isPrime[n];
    int primes[n], primesSize = 0;
    memset(isPrime, 0, sizeof(isPrime));

    for (int i = 2; i < n; ++i) {
        if (!isPrime[i]) {
            primes[primesSize++] = i;
        }
        for (int j = 0; j < primesSize && i * primes[j] < n; ++j) {
            isPrime[i * primes[j]] = 1;
            if (i % primes[j] == 0) {
                break;
            }
        }
    }
    return primesSize;
}// 时间复杂度 O(n), 空间复杂度 O(n)



// 10. 完全平方数
// 题目描述：给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
//      （完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。）
//  一般思路：
//  数学解法：四平方和定理 时间复杂度O(N ^ (1/2) ) https://leetcode.cn/problems/perfect-squares/solution/wan-quan-ping-fang-shu-by-leetcode-solut-t99c/
//
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

// 判断是否为完全平方数
bool isPerfectSquare(int x) {
    int y = sqrt(x);
    return y * y == x;
}
// 判断是否能表示为 4^k*(8m+7)
bool checkAnswer4(int x) {
    while (x % 4 == 0) {
        x /= 4;
    }
    return x % 8 == 7;
}
int numSquares2(int n) {
    if (isPerfectSquare(n)) {
        return 1;
    }
    if (checkAnswer4(n)) {
        return 4;
    }
    for (int i = 1; i * i <= n; i++) {
        int j = n - i * i;
        if (isPerfectSquare(j)) {
            return 2;
        }
    }
    return 3;
}


// 11. 按权重生成随机数
class Solution {
    vector<int> W;
public:
    Solution(vector<int>& w) {
        W.push_back(w[0]);
        for(int i = 1; i < w.size(); ++i) {
            W.push_back(W.back() + w[i]);
        }
    }

    int pickIndex() {
        int weight = rand() % W.back();
        return upper_bound(W.begin(), W.end(), weight) - W.begin();
    }
};


// 12. 圆内随机生成点
// 拒绝采样的意思是说：我们在一个更大的范围内生成随机数，并拒绝掉那些不在题目给定范围内的随机数，此时保留下来的随机数都是在范围内的。
class SolutionGenPointInCircle {
private:
    mt19937 gen{random_device{}()}; // mt19937 引擎
    uniform_real_distribution<double> dis; // 生成均匀分布
    double xc, yc, r;

public:
    SolutionGenPointInCircle(double radius, double x_center, double y_center): dis(-radius, radius), xc(x_center), yc(y_center), r(radius) {}

    vector<double> randPoint() {
        while (true) {
            double x = dis(gen), y = dis(gen);
            if (x * x + y * y <= r * r) {
                return {xc + x, yc + y};
            }
        }
    }
};

class SolutionGenPointInCircleV2 {
private:
    mt19937 gen{random_device{}()};
    uniform_real_distribution<double> dis;
    double xc, yc, r;

public:
    SolutionGenPointInCircleV2(double radius, double x_center, double y_center): dis(0, 1), xc(x_center), yc(y_center), r(radius) {}

    vector<double> randPoint() {
        double u = dis(gen), theta = dis(gen) * 2 * acos(-1.0);
        double r = sqrt(u);
        return {xc + r * cos(theta) * this->r, yc + r * sin(theta) * this->r};
    }
};

// 13. 非重叠矩形中随机生成点
// 题目描述：坐标中有n个非重叠的矩形，请在矩形中均匀随机的生成点
// 思路： 先随机的选一个矩形，然后在矩形内部随机生成点。
//   选矩形可以根据面积大小来，选好后矩形内生成点很容易
// 实际实现时可以使用前缀和+二分 来确定选中的矩形
class SolutionGenPointInRect {
public:
    int n;
    vector<vector<int>> rects;
    vector<int> arr;
    SolutionGenPointInRect(vector<vector<int>>& _rects) {
        rects = _rects;
        n = rects.size();
        arr.push_back(0);
        for(auto& rect : rects) {
            int m = rect[2] - rect[0] + 1;
            int n = rect[3] - rect[1] + 1;
            arr.push_back(arr.back()+m*n);
        }
    }

    vector<int> pick() {
        int k = rand() % arr.back() + 1;
        int left = 0, right = n;
        while(left < right) {
            int mid =  (left + right)/2;
            if(arr[mid] >= k) right = mid;
            else left = mid + 1;
        }
        vector<int> target = rects[right - 1];
        int x = target[2] - target[0] + 1;
        int y = target[3] - target[1] + 1;
        return {rand()%x+target[0], rand()%y+target[1]};
    }
};


// 14. 分发糖果
// 题目描述：n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。
//  你需要按照以下要求，给这些孩子分发糖果：
//      - 每个孩子至少分配到 1 个糖果。
//      - 相邻两个孩子评分更高的孩子会获得更多的糖果。
//  请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。
// 思路：换一种思路理解规则
//     左规则：当ratings[i−1] < ratings[i] 时，i 号学生的糖果数量将比 i−1 号孩子的糖果数量多。
//     右规则：当ratings[i] > ratings[i+1] 时，i 号学生的糖果数量将比 i+1 号孩子的糖果数量多
// 我们遍历该数组两次，处理出每一个学生分别满足左规则或右规则时，最少需要被分得的糖果数量。每个人最终分得的糖果数量即为这两个数量的最大值。
int candy(vector<int>& ratings) {
    int n = ratings.size();
    vector<int> l(n);
    for (int i = 0; i < n; i++) {
        if (i > 0 && ratings[i] > ratings[i - 1]) {
            l[i] = l[i - 1] + 1;
        } else {
            l[i] = 1;
        }
    }
    int right = 0, ret = 0;
    for (int i = n - 1; i >= 0; i--) {
        if (i < n - 1 && ratings[i] > ratings[i + 1]) {
            right++;
        } else {
            right = 1;
        }
        ret += max(l[i], right);
    }
    return ret;
} // 时间复杂度：O(n)
// 空间复杂度：O(n)

// 15. 计算数字序列中某一位的数字
// 题目描述：数字以0123456789101112131415…的格式序列化到一个字符序列中。求在这个序列中任意第n位对应的数字
// https://leetcode.cn/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/solution/mian-shi-ti-44-shu-zi-xu-lie-zhong-mou-yi-wei-de-6/
int findNthDigit(int n) {
    int digit = 1;
    long start = 1;
    long count = 9;

    while(n > count) { // 1.确定所求数位的所在数字的位数
        n = n - count;
        digit += 1;
        start *= 10;
        count = digit * start * 9;
    }

    long num = start + (n - 1) / digit; // 2.确定所求数位所在的数字（所求数位在从数字 start开始的第 (n−1)/digit 个数字 中，start 为第 0 个数字）
    count = (n-1)%digit; // 计算是该数字第几位
    //        //举个例子，9是第9位，10~99共180位，189位就是99中的后一个9，
    //        //那么按程序走，189-9=180=n，不进入循环，所以start=10，(180-1)/2=89余1，
    //        //这180位是从start前一位(9)开始算的，所以要-1，否则就是从start(10)中的1开始加了
    string s=to_string(num);
    int ans=s[count]-'0';
    return ans;
}





#endif //DATASTRUCT_ALGORITHM_MATH_PROBLEM_H

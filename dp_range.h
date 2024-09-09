//
// Created by kai.chen on 2022/2/7.
//
// 区间dp：把问题锁定在一个区间，大区间的答案是否可以由小区间的结论推出答案
//
//      1. 统计不同的回文子序列
//        1.1. 最长回文子序列
//
//      2. 石子归并
//      3. 猜数字大小II
#ifndef DATASTRUCT_ALGORITHM_DP_RANGE_H
#define DATASTRUCT_ALGORITHM_DP_RANGE_H

// 1. 统计不同的回文子序列 https://leetcode.cn/problems/count-different-palindromic-subsequences/solution/tong-ji-butong-by-jiang-hui-4-q5xf/
// 题目描述：统计不同的回文子序列， s[i] 仅包含 'a', 'b', 'c' 或 'd'
// 思路： 往长度较少的回文串两端添加字符，可能组成新的长度大的回文串，容易想到「区间 DP」
//  区间 DP 的一般思路，定义 f[i][j] 为考虑字符串 s 中的[i,j] 范围内回文子序列的个数，最终答案为 f[0][n−1]。
// 通过枚举 abcd 作为回文方案「边缘字符」来进行统计，即分别统计各类字符作为「边缘字符」时对 f[i][j] 的贡献，此类统计方式天生不存在重复性问题。
// 假设当前枚举到的字符为 k ：
//      若 s[i...j] 中没有字符 k，则字符 k 对 f[i][j] 贡献为 00，跳过；
//      若 s[i...j]中存在字符 k，根据字符 k 在范围s[i...j] 中「最小下标」和「最大下标」进行分情况讨论，假设字符 k 在s[i...j] 中「最靠左」的位置为 l，「最靠右」的位置为 r：
//          当l=r 时，此时字符 k 对 f[i][j] 的贡献为 1，即 k 本身；
//          当l=r−1 时，说明字符 k中间不存在任何字符，此时字符 k 对f[i][j] 的贡献为 2，包括 k 和 kk 两种回文方案；
//          其余情况，可根据已算得的「小区间回文方案」进行延伸（两段分别补充位于 l 和 r 的字符 k），得到新的大区间方案，此部分对f[i][j] 的贡献是f[l+1][r−1]，另外还有单独的子序列只用 k 和 kk 两种回文方案，因此总的对答案的贡献为 f[l+1][r−1]+2。
int countPalindromicSubsequences(string s) {
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));
    vector<int> left(4, -1), right;
    for(int i = n - 1; i >= 0; i--) {
        left[s[i] - 'a'] = i;

        right = vector<int>(4, -1);
        for(int j = i; j < n; j++) {
            right[s[j] - 'a'] = j;

            for(int k = 0; k < 4; k++) {
                if(left[k] == -1 || right[k] == -1)
                    continue;
                if(left[k] == right[k]) {
                    dp[i][j] = (dp[i][j] + 1) % mod;
                }else if(left[k] + 1 == right[k]) {
                    dp[i][j] = (dp[i][j] + 2) % mod;
                }else {
                    dp[i][j] = (dp[i][j] + dp[left[k] + 1][right[k] - 1] + 2) % mod;
                }
            }
        }
    }
    return dp[0][n - 1];
}// 时间复杂度：O(C×n^2)，其中 C=4 为字符集大小
// 空间复杂度：O(n^2)


// 2. 涂色
// https://www.luogu.com.cn/problem/solution/P4170
// 题意是求对字符串的最少染色次数，设f[i][j]为字符串的子串s[i]~s[j]的最少染色次数，我们分析一下：
//
//当i==j时，子串明显只需要涂色一次，于是f[i][j]=1。
//
//当i!=j且s[i]==s[j]时，可以想到只需要在首次涂色时多涂一格最前面/最后面即可，于是f[i][j]=min(f[i][j-1],f[i+1][j])
//
//当i!=j且s[i]!=s[j]时，我们需要考虑将子串断成两部分来涂色，于是需要枚举子串的断点，设断点为k，那么f[i][j]=min(f[i][j],f[i][k]+f[k+1][j])
#include<cstring>
#include<algorithm>
using namespace std;
char s[52];
int f[52][52];
int main() {
    int n;
    scanf("%s",s+1);
    n=strlen(s+1);
    memset(f,0x7F,sizeof(f));		//由于求最小，于是应初始化为大数
    for(int i=1;i<=n;++i){
        f[i][i]=1;
    }
    for(int l=1;l<n;++l){
        for(int i=1,j=1+l;j<=n;++i,++j) {
            if(s[i]==s[j]){
                f[i][j]=min(f[i+1][j],f[i][j-1]);
            }else {
                for (int k = i; k < j; ++k) {
                    f[i][j] = min(f[i][j], f[i][k] + f[k + 1][j]);
                }
            }
        }
    }
    printf("%d",f[1][n]);
    return 0;
}


#endif //DATASTRUCT_ALGORITHM_DP_RANGE_H
// https://www.bilibili.com/video/BV1i7411i74r?spm_id_from=333.999.0.0&vd_source=0b16c952d9a60a452d6365bb2d227f4a
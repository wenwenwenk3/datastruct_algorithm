//
// Created by kai.chen on 2021/12/12.
//      1. LCS: 最长公共子序列长度 (Longest Common Subsequence)
//          变体1.1：最长公共子串（Longest Common Substring）
//      2. 编辑距离: 将字符串s1转换成s2最少需要多少次操作
//      3. 最长回文子序列
//          变体3.1：最长回文子串（Longest Palindrome Substring）
//      4. 以最少插入次数构造回文串

//      5. 正则表达式匹配  -> 见 ip_address_op.h
//
//      1.2. 最长公共前缀
//      1.3 验证回文串，变体：最多删除一个字符，判断是否能成为回文字符串。
//
//      1.4 字符串中 回文子串 的数目
//      1.5 不同的子序列数目
#ifndef DATASTRUCT_ALGORITHM_DP_2D_H
#define DATASTRUCT_ALGORITHM_DP_2D_H
#include <cmath>

// 二维dp
// 1. LCS: 最长公共子序列长度 (Longest Common Subsequence)
int longestCommonSubsequence(string str1, string str2){
    int m = str1.size(), n = str2.size();
    if(m *n == 0){
        return 0;
    }
    int dp[m+1][n+1];
    for(int i = 0; i <= m; i++) dp[i][0] = 0;
    for(int j = 0; j <= n; j++) dp[0][j] = 0;

    for(int i = 1; i <= m; i++){
        for(int j = 1; j <= n; j++){
            if(str1[i-1] == str2[j-1]){
                dp[i][j] = dp[i-1][j-1] + 1;
            }
            else{
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    return dp[m][n];
}

void test_t1(){
    int cnt = longestCommonSubsequence("abche", "auytce");
    cout<<"longest common subseqence:"<<cnt;
}

// 变体1.1：最长公共子串（Longest Common Substring）
//dp[i][j] 表示以i，j结尾的最长公共子串长度
// 计算dp[i][j] 的方法如下：
// base case
//矩阵 dp 的第一列 dp[0…m-1][0].对于 某个位置（i，0）如果str1[i]==str2[0],则dp[i][0]=1,否则dp[i][0]=0
//矩阵 dp 的第一行 dp[0][0…n-1].对于 某个位置（0，j）如果str1[0]==str2[j],则dp[0][j]=1,否则dp[0][j]=0
//其他位置从左到右从上到下计算，dp[i][j]的值只有两种情况：
//1). str1[i]==str2[j],dp[i][j]=dp[i-1][j-1]+1;
//
//2). str1[i]!=str2[j]则dp[i][j]=0;
int longestCommonSubstring(string str1, string str2){
    int m = str1.size(), n = str2.size();
    if(m *n == 0){
        return 0;
    }
    int dp[m][n];
    for(int i = 0; i < m; i++) dp[i][0] = str1[i] == str2[0] ? 1: 0;
    for(int j = 0; j < n; j++) dp[0][j] = str2[j] == str1[0] ? 1: 0;

    for(int i = 1; i < m; i++){
        for(int j = 1; j < n; j++){
            if(str1[i] == str2[j]){
                dp[i][j] = dp[i-1][j-1] + 1;
            }
            else{
                dp[i][j] = 0;
            }
        }
    }

    int maxlen = dp[0][0];
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            maxlen = max(maxlen, dp[i][j]);
        }
    }

    return maxlen;
}

// 2. 编辑距离: 将字符串s1转换成s2最少需要多少次操作
//
int minDistance(string s1, string s2){
    int m = s1.size(), n = s2.size();
    if(m * n == 0) return m+n;

    int dp[m+1][n+1];
    for(int i = 0; i <=m; i++) dp[i][0] = i;
    for(int j = 0; j <=n; j++) dp[0][j] = j;

    for(int i = 1; i < m+1; i++){
        for(int j = 1; j < n+1; j++){
            if(s1[i-1] == s2[j-1]){
                dp[i][j] = dp[i-1][j-1];
            }
            else{
                dp[i][j] = min(dp[i-1][j]+1, min(dp[i][j-1]+1, dp[i-1][j-1]+1));
            }
        }
    }
    return dp[m][n];
}


// 3. 最长回文字序列
// 记住：dp数组定义为，子串s[i..j]中，最长回文子序列的长度为dp[i][j]
// 根据定义要求的结果就是dp[0][n-1]
//1). str[i]==str[j],dp[i][j]=dp[i+1][j-1]+2;
//
//2). str[i]!=str[j]则dp[i][j]= max(dp[i+1][j], dp[i][j-1]);
int longestPalindromeSubseq(string s){
    int n = s.size();
    if(n == 0) return 0;
    vector<vector<int>> dp(n, vector<int>(n,0));
    for(int i = 0; i < n; i++){ // base case
        dp[i][i] = 1;
    }
    // 斜着遍历，从下到上从左到右
    for(int i = n-2; i >= 0; i--){
        for(int j = i+1; j < n; j++){
            if(s[i]==s[j]){
                dp[i][j] = dp[i+1][j-1]+2;
            }
            else{
                dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
            }
        }
    }
    return dp[0][n-1];
}
// 变体3.1：最长回文子串（Longest Palindrome Substring）
// (1). 中心扩展：首先思考：是否可以将s反过来,找s'和s的最长公共子串。这种思路看起来可以但实际上不行
//          比如s=abacd, s'=dcaba 答案确实是aba，但当s= aacxyzcaa, s'= aaczxycaa 结果是aac,但答案应该是aa
//      正确思路：
//          寻找回文串的思想应该是从中间开始扩散
//             可以设计一个辅助函数寻找以i为中心的最长回文子串，然后遍历字符串，若长度大于之前的就更新为新的最长子串
// (2). 动态规划：dp[i][j]=1表示从i到j是回文串，dp(i,j)==dp(i+1,j−1)∧(S[i] == S[j]
// (3). 马拉车(Manacher): 知道它是O(n)即可，中间加#，两侧加^G，看了下特别复杂（面试非要你默出来的话，可能是想为难你）

// 辅助函数：计算以s[i]和s[j]为中心的最长回文串。当l和r相等时找的是奇数
string palindrome(string& s, int n, int l ,int r){
    while(l >= 0 && r <= n-1 && s[l]==s[r]){
        l--;
        r++;
    }
    l++,r--; // while退出条件的时候l和r都多移动了一位
    return s.substr(l,r-l+1);
}
string longestPalindrome(string s){
    string res;
    int n = s.size();
    if(n <= 1) return s;
    for(int i = 1; i < n; i++){
        string s1 = palindrome(s, n, i-1, i);
        string s2 = palindrome(s, n, i, i);
        // res = max(res, s1, s2);
        if(s1.size() > res.size()) res = s1;
        if(s2.size() > res.size()) res = s2;
    }
    return res;
} // 时间复杂度O(n^2), 空间复杂度O(1)

string longestPalindrome_dp(string s) {
    int n = s.size();
    if(n <= 1) return s;
    int start = 0; //回文串起始位置
    int maxlen = 1; //回文串最大长度
    vector<vector<int>>  dp(n,vector<int>(n, 0)); //定义二维动态数组
    // base case
    for(int i=0;i<n;i++){
        dp[i][i]=1; // 单个字符是回文串
        if(i < n-1 && s[i]==s[i+1]){ // 连续两个相同字符是回文串
            dp[i][i+1] = 1; // 标记为合法回文串
            maxlen = 2; // 更新
            start = i;
        }
    }
    //l表示检索的子串长度，等于3表示先检索长度为3的子串
    for(int l = 3; l <= n; l++){
        for(int i = 0; i+l-1 < n; i++){
            int j = l+i-1;//终止字符位置
            if(s[i]==s[j] && dp[i+1][j-1]==1){ //状态转移
                dp[i][j] = 1;  //标记为合法回文串
                start = i;
                maxlen = l;
            }
        }
    }
    return s.substr(start,maxlen);//获取最长回文子串
} // 时间复杂度O(n^2), 空间复杂度O(n)




// 4.以最少插入次数构造回文串
// dp数组定义：对于s[i..j], 最少需要进行dp[i][j]次插入才能变成回文串, 目标是dp[0][n-1]
// 1). s[i]==s[j] dp[i][j] = dp[i+1][j-1]
// 思考 2). s[i]!=s[j] dp[i][j] = dp[i+1][j-1]+2 ? （不一定需要插2次
// 正确 即做选择，现将s[i+1][j]或者dp[i][j-1]变成回文串,再插入一个字符就好了
// 2). s[i]!=s[j] dp[i][j] = min(dp[i+1][j], dp[i][j-1])+1 ?
int minIntersection(string s){
    int n = s.size();
    if(n == 0) return 0;
    vector<vector<int>> dp(n, vector<int>(n,0));
//    for(int i = 0; i < n; i++){ // base case
//        dp[i][i] = 0;
//    }
    for(int i = n-2; i >= 0; i--){    // 斜着遍历，从下到上从左到右
        for(int j = i+1; j < n; j++){
            if(s[i] == s[j]){
                dp[i][j] = dp[i+1][j-1];
            }
            else{
                dp[i][j] = min(dp[i+1][j], dp[i][j-1])+1;
            }
        }
    }
    return dp[0][n-1];
}


// 5.正则表达式匹配  -> 见 ip_address_op.h


//




#endif //DATASTRUCT_ALGORITHM_DP_2D_H



// 1.2. 最长公共前缀
// 求字符串数组的最长公共前缀
//  思路，以第一个字符串为基准，纵向匹配
string longestCommonPrefix(vector<string>& strs) {
    if (strs.size() == 0) {
        return "";
    }
    int length = strs[0].size();
    int count = strs.size();
    for (int i = 0; i < length; ++i) {
        char c = strs[0][i];
        for (int j = 1; j < count; ++j) {
            if (i == strs[j].size() || strs[j][i] != c) {
                return strs[0].substr(0, i);
            }
        }
    }
    return strs[0];
}

// 1.3 验证回文串
// 题目描述：
//  输入: "A man, a plan, a canal: Panama"
//  输出: true  解释："amanaplanacanalpanama" 是回文串
// 思路：
//    预处理筛选出字符，再翻转字符，判断是否相等
//          或者用前后两个指针判断回文
bool isPalindrome(string s) {
    string sgood;
    for (char ch: s) {
        if (isalnum(ch)) {
            sgood += tolower(ch);
        }
    }
    string sgood_rev(sgood.rbegin(), sgood.rend());
    return sgood == sgood_rev;
}

bool isPalindrome2(string s) {
    string sgood;
    for (char ch: s) {
        if (isalnum(ch)) {
            sgood += tolower(ch);
        }
    }
    int n = sgood.size();
    int left = 0, right = n - 1;
    while (left < right) {
        if (sgood[left] != sgood[right]) {
            return false;
        }
        ++left;
        --right;
    }
    return true;
}

// 1.3 变体：最多删除一个字符，判断是否能成为回文字符串。
bool checkPalindrome(const string& s, int left, int right) {
    for (int i = left, j = right; i < j; ++i, --j) {
        if (s[i] != s[j]) {
            return false;
        }
    }
    return true;
}
bool validPalindrome(string s) {
    int n = s.size();
    int left = 0, right = n - 1;
    while (left < right) {
        if (s[left] != s[right]) {
            return checkPalindrome(s, left, right - 1) || checkPalindrome(s, left + 1, right);
        }
        ++left;
        --right;
    }
    return true;
}

// 1.5 不同的子序列数目
// 题目描述：给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
// 思路：对于两个字符串匹配，一个非常通用的dp状态定义如下：
// 定义 f[i][j] 为考虑 s 中 [0,i] 个字符，t 中 [0,j] 个字符的匹配个数。
//  当s[i] != t[j]时: dp[i][j] = dp[i - 1][j];
//  当s[i] == t[j]时: dp[i][j] = dp[i - 1][j] + dp[i - 1][j-1];
int numDistinct(string s, string t) {
    int n = s.size(), m = t.size();
    // 技巧：往原字符头部插入空格，这样得到 char 数组是从 1 开始
    // 同时由于往头部插入相同的（不存在的）字符，不会对结果造成影响，而且可以使得 f[i][0] = 1，可以将 1 这个结果滚动下去
    s = " " + s;
    t = " " + t;
    // dp[i][j]
    vector<vector<int>> dp(n + 1,vector<int>(m + 1,0));
    for(int i = 0; i < n; i++) dp[i][0] = 1;
    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= m; j++){
            // 包含两种决策：
            // 不使用 cs[i] 进行匹配，则有 dp[i][j] = dp[i - 1][j]
            dp[i][j] = dp[i - 1][j];
            // 使用 cs[i] 进行匹配，则要求 cs[i] == ct[j]，然后有dp[i][j] += dp[i - 1][j - 1]
            if(s[i] == t[j]) {
                // 注意int中间结果可能越界需要转成long long，又题目明确答案范围小于INT_MAX所以这里可以取余
                dp[i][j] = (0LL + dp[i][j] + dp[i - 1][j - 1]) % INT_MAX;
            }
        }
    }
    return dp[n][m];
}// 时间复杂度：O(n * m) 空间复杂度：O(n * m)
//作者：AC_OIer 链接：https://leetcode-cn.com/problems/distinct-subsequences/solution/xiang-jie-zi-fu-chuan-pi-pei-wen-ti-de-t-wdtk/

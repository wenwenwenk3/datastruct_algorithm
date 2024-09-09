//
// Created by kai.chen on 2021/12/19.
//
//      1. 反转字符串里的单词
//      2. 单词拆分
//
#ifndef DATASTRUCT_ALGORITHM_STRING_REVERSE_H
#define DATASTRUCT_ALGORITHM_STRING_REVERSE_H
#include <deque>
#include <string>
using namespace std;

// 1. 反转字符串里的单词
// 题目描述：输入：s = "the sky is blue" 输出："blue is sky the"
// 思路1： 先反转整个字符串，再挨个单词进行反转
string reverseWords(string s) {
    // 反转整个字符串
    reverse(s.begin(), s.end());

    int n = s.size();
    int idx = 0;
    for (int start = 0; start < n; ++start) {
        if (s[start] != ' ') {
            // 填一个空白字符然后将idx移动到下一个单词的开头位置
            if (idx != 0) s[idx++] = ' ';

            // 循环遍历至单词的末尾
            int end = start;
            while (end < n && s[end] != ' ') s[idx++] = s[end++];

            // 反转整个单词
            reverse(s.begin() + idx - (end - start), s.begin() + idx);

            // 更新start，去找下一个单词
            start = end;
        }
    }
    s.erase(s.begin() + idx, s.end());
    return s;
} // 时间复杂度O(N),空间复杂度O(1),1存放变量

// 思路2：借助双端队列的特性，可以依次进行头插，取出每一个反转的单词
string reverseWords(string s) {
    int start = 0, end = s.size() - 1;
//    // 去掉字符串开头的空白字符
//    while (start <= end && s[start] == ' ') ++start;
//    // 去掉字符串末尾的空白字符
//    while (start <= end && s[end] == ' ') --end;
    deque<string> deq;
    string word;
    for(; start <= end; start++) {
        char c = s[start];
        if(c != ' '){
            word += c;
        }
        else if (c == ' ' && !word.empty()) {
            // 将单词 push 到队列的头部
            deq.push_front(move(word));
            word = "";
        }
    }
    deq.push_front(move(word));

    string ans;
    while (!deq.empty()) {
        ans += deq.front();
        deq.pop_front();
        if (!deq.empty()) ans += ' ';
    }
    return ans;
} // 时间复杂度O(N), 空间复杂度O(N)

// k个字符翻转字符串
string reverseStrKGroup(string s, int k) {
    int n = s.length();
    for (int i = 0; i < n; i += 2 * k) {
        reverse(s.begin() + i, s.begin() + min(i + k, n));
    }
    return s;
}




// 2.单词拆分
// 题目描述：给你一个字符串 s 和一个字符串列表 wordDict 作为字典，判定 s 是否可以由空格拆分为一个或多个在字典中出现的单词。
// 思路：对于前i个字符组成的s[0..i-1], 需要枚举 s[0..i−1] 中的分割点 j ，看 s[0..j−1] s[j..i-1]组成的字符串是否都合法
//      定义dp[i]表示前i个字符是否合法
//   则dp转移方程为： dp[i] = dp[j] && isValid(s[j..i-1])
//   base case: dp[0] = true
bool wordBreak(string s, vector<string>& wordDict) {
    auto wordDictSet = unordered_set <string> (); // 用set来快速判断s[j..i-1]合法性
    for (auto word: wordDict) {
        wordDictSet.insert(word);
    }

    vector<bool> dp(s.size() + 1);
    dp[0] = true;
    for (int i = 1; i <= s.size(); ++i) {
        // 遍历分割点j
        for (int j = 0; j < i; ++j) {
            // 合法判断 dp[i] = dp[j] && isValid(s[j..i-1])
            if (dp[j] && wordDictSet.find(s.substr(j, i - j)) != wordDictSet.end()) {
                dp[i] = true;
                break; // 前i个字符是合法的
            }
        }
    }

    return dp[s.size()];
}
// 时间复杂度：O(N^2) dp有n个状态需要计算，每次计算需要枚举O(n)个分割点
// 空间复杂度：O(N) 需要一个长度为N的dp表，和hash_set

// 2.单词拆分II
// 题目描述：给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，
//      在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。
//   输入:s = "catsanddog" wordDict = ["cat", "cats", "and", "sand", "dog"]
//   输出: ["cats and dog", "cat sand dog"]
// 思路：没辙这题太难，劝退题

#endif //DATASTRUCT_ALGORITHM_STRING_REVERSE_H

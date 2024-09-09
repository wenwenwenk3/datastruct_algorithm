//
// Created by kai.chen on 2022/4/10.
//
// 1. 删除注释
// 2. 独特的电子邮件地址

#ifndef DATASTRUCT_ALGORITHM_STRING_OP_H
#define DATASTRUCT_ALGORITHM_STRING_OP_H

// 1. 删除注释
vector<string> removeComments(vector<string>& source) {
    bool isComment = false;
    vector<string> res;
    string curr = "";
    for (string s : source){
        // cout << s << endl;
        int i = 0;
        if (!isComment){
            curr = "";
        }
        int n = s.size();
        while (i < n){
            if (!isComment && i < n-1 && s[i] == '/' && s[i+1] == '*'){
                isComment = true;
                // 继续找下去,这里实际要+2，但是后面会+1
                ++i;
            }else if (isComment && i < n-1 && s[i] == '*' && s[i+1] == '/'){
                isComment = false;
                // 继续找下去,这里实际要+2，但是后面会+1
                ++i;
            }else if (!isComment && i < n-1 && s[i] == '/' && s[i+1] == '/'){
                break;
            }else if (!isComment){
                curr += s[i];
                // cout << curr<< " " << endl;
            }
            ++i;
        }
        if (!isComment && !curr.empty()){
            res.push_back(curr);
        }
    }
    return res;
}

// 2. 独特的电子邮件地址
int numUniqueEmails(vector<string>& emails) {
    unordered_set<string> emailSet;
    for (auto &email: emails) {
        string local;
        for (char c: email) {
            if (c == '+' || c == '@') {
                break;
            }
            if (c != '.') {
                local += c;
            }
        }
        emailSet.emplace(local + email.substr(email.find('@')));
    }
    return emailSet.size();
}


// 匹配子序列的单词数
// 首先题目给出字符串 s 和一个字符串数组 words，我们需要统计字符串数组中有多少个字符串是字符串 s 的子序列。
//那么最朴素的方法就是我们对于字符串数组 words 中的每一个字符串和字符串 s 尝试进行匹配，
// 我们可以用「双指针」的方法来进行匹配——用 i 指向字符串 s 当前遍历到的字符，j 指向当前需要匹配的字符串 t 需要匹配的字符，
// 初始 i = 0, j=0，如果 s[i]=t[j] 那么将指针 i 和 j 同时往后移动一个单位，否则仅 i 移动 i 往后移动一个单位，
// 并在 i 指向字符串 s 结尾或者 j 指向 t 结尾时结束匹配过程，然后判断 j 是否指向 t 的结尾，若指向结尾则说明 t 为字符串 s 的子序列，否则不是。
//
int numMatchingSubseq(string s, vector<string> &words) {
    vector<queue<pair<int, int>>> queues(26);
    for (int i = 0; i < words.size(); ++i) {
        queues[words[i][0] - 'a'].emplace(i, 0);
    }
    int res = 0;
    for (char c : s) {
        auto &q = queues[c - 'a'];
        int size = q.size();
        while (size--) {
            auto [i, j] = q.front();
            q.pop();
            ++j;
            if (j == words[i].size()) {
                ++res;
            } else {
                queues[words[i][j] - 'a'].emplace(i, j);
            }
        }
    }
    return res;
}


#endif //DATASTRUCT_ALGORITHM_STRING_OP_H

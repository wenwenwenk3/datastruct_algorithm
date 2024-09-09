//
// Created by kai.chen on 2021/11/29.
//  1. 括号匹配
//      1.1 最少的修改次数让括号匹配
//      1.1.1 最少的移除让括号匹配
//      1.2 判断有效的括号字符串 包含*
//      1.3 有效括号的最大嵌套深度
//      1.4 括号的分数
//  2. 括号生成
//  3. 最长有效括号长度
//  4. 有效括号的嵌套深度 - 分配两个字符串让深度最小
//  5. 删除无效的括号
//  6. 翻转每队括号间的子串

#ifndef DATASTRUCT_ALGORITHM_KUOHAO_MATCH_H
#define DATASTRUCT_ALGORITHM_KUOHAO_MATCH_H
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <vector>
using namespace std;

// 1.判断括号是否匹配
//   每次判断如果是左括号：入栈，如果是右括号：就判断栈里的左括号是否和它匹配，如果匹配就出栈一个左括号
//   如果最后栈不为空，或者栈在循环结束前就为空
bool isValidKuoHao(string s){
    int n = s.size();
    if(n%2 != 0){
        return false;
    }

    unordered_map<char, char> khmap = {
            {')', '('},
            {'}', '{'},
            {']', '['}
    };
    stack<char> stk;
    for(auto i: s){
        if(khmap.count(i)){  //judge cur i is ')/}/]'
            if(stk.empty() || stk.top() != khmap.at(i)){
                return false;
            }
            stk.pop();
        }
        else {
            stk.push(i);
        }
    }

    return stk.empty();
}

// 1.1 最少的修改次数让括号匹配
// 题目描述：给你一个只含有括号的字符串，你可以将一种类型的左括号改成另外一种类型，右括号改成另外一种右括号
//问你最少修改多少次，才能使得这个字符串匹配，输出次数
// 思路：
//  每次判断，如果遇上左括号，就加进栈里。如果遇上右括号，就判断栈里的左括号是否和它匹配，不匹配就加一。不论匹不匹配，判断后都要让左括号出栈。
//如果最后栈不为空，或者栈在循环结束前就为空，那么不论怎么改变，左右括号都不可能刚好匹配。
int MinStepToValidKuoHao(string& s){
    int n = s.size();
    if(n%2 != 0){
        return -1;
    }

    unordered_map<char, char> khmap = {
            {')', '('},
            {'}', '{'},
            {']', '['}
    };
    stack<char> stk;
    int step = 0;
    for(auto i: s){
        if(khmap.count(i)){  //judge cur i is ')/}/]'
            if(stk.empty()){
                return -1;
            }
            if(stk.top() != khmap.at(i)){
                step++;
            }
            stk.pop();
        }
        else {
            stk.push(i);
        }
    }
    return stk.empty()? -1: step;
}

// 1.1 最少的移除让括号匹配
// 题目描述： 给你一个由 '('、')' 和小写字母组成的字符串 s
//      你需要从字符串中删除最少数目的 '(' 或者 ')' （可以删除任意位置的括号)，使得剩下的「括号字符串」有效。
// 例如 输入：s = "lee(t(c)o)de)"
//  输出："lee(t(c)o)de"， 解释："lee(t(co)de)" , "lee(t(c)ode)" 也是一个可行答案。
//
// 思路： (1).向前遍历-统计无效左括号，(2)向后遍历-统计无效右括号，(3).最后替换无效括号
string minRemoveToMakeValid(string s) {
    int lcount = 0, rcount = 0;
    for(auto & c : s){
        if(c == '('){
            lcount++;
        }else if(c == ')'){
            lcount--;
            if(lcount < 0){ // 出现不合法的 )
                c = '0';
                lcount = 0; //重新计算
            }
        }
    }
    for(int i = s.length()-1; i>=0; i--){
        if(s[i] == ')'){
            rcount++;
        }else if(s[i] == '('){
            rcount--;
            if(rcount < 0){ // 出现不合法的 )
                s[i] = '0';
                rcount = 0; //重新计算
            }
        }
    }
    string res;
    for(const auto& c : s){
        if(c !='0'){
            res+=c;
        }
    }
    return res;
}


// 1.2 判断有效的括号字符串 包含*
// 贪心思路：
// 从左到右遍历字符串，遍历过程中，考虑未匹配的左括号数量的变化：
//      如果遇到左括号，则未匹配的左括号数量加 1；
//      如果遇到右括号，则需要有一个左括号和右括号匹配，因此未匹配的左括号数量减 1；
//      如果遇到星号，由于星号可以看成左括号、右括号或空字符串，因此未匹配的左括号数量可能加 1、减 1 或不变。
// 基于上述分析，可以在遍历过程中维护未匹配的左括号数量可能的最小值和最大值
//  如果遇到左括号，则将最小值和最大值分别加 1；
//  如果遇到右括号，则将最小值和最大值分别减 1；
//  如果遇到星号，则将最小值减 1，将最大值加 1。
// 最后注意范围：未匹配的左括号数量必须非负，因此当最大值变成负数时，说明没有左括号可以和右括号匹配，直接返回 false。
//      当最小值为 0 时，不应将最小值继续减少，以确保最小值非负。
bool checkValidString(string s) {
    int minCount = 0, maxCount = 0;
    int n = s.size();
    for (int i = 0; i < n; i++) {
        char c = s[i];
        // 如果遇到左括号，则将最小值和最大值分别加 1；
        if (c == '(') {
            minCount++;
            maxCount++;
        }
        // 如果遇到右括号，则将最小值和最大值分别减 1；
        else if (c == ')') {
            minCount = max(minCount - 1, 0);
            maxCount--;
            if (maxCount < 0) {
                return false;
            }
        }
        // 如果遇到星号，则将最小值减 1，将最大值加 1。
        else {
            minCount--;
            if(minCount < 0){
                minCount = 0;
            }
            maxCount++;
        }
    }
    // 遍历结束时，所有的左括号都应和右括号匹配，因此只有当最小值为 0 时，字符串 s 才是有效的括号字符串。
    return minCount == 0;
}

// 1.3 有效括号的最大嵌套深度
int maxDepth(string s) {
    int ans = 0, size = 0; // 只需要考虑栈的大小，可以直接用一个变量 size 表示栈的大小
    for (char ch : s) {
        if (ch == '(') {
            size++;
            ans = max(ans, size);
        }
        else if (ch == ')') {
            size--;
        }
    }
    return ans;
}

// 1.4 括号的分数
// () 得 1 分。
// AB 得 A + B 分，其中 A 和 B 是平衡括号字符串。
// (A) 得 2 * A 分，其中 A 是平衡括号字符串。
int scoreOfParentheses(string s) {
    int sum = 0;
    int stksize = 0;
    for (int i = 0; i < s.length(); i++) {
        if (s[i] == '(') {
            stksize++;
        } else {
            stksize--;
            if (s[i-1] == '('){//避免处于最后面的)的重复计算
                int xsum = 1;
                for(int j=0;j<stksize;j++)
                    xsum = xsum*2;
                sum += xsum;
            }
        }
    }

    return sum;
}


// 2. 括号生成
// (left, right 表示当前左括号和右括号剩余的数量)
void generateKuoHao_backtrack(int left, int right, string& track, vector<string>& res){
    if(left < 0 || right < 0) return ;
    if(left > right) return ; // 左括号剩下的多
    if(left == 0 && right == 0){
        res.push_back(track);
        return;
    }

    track.push_back('('); // 做选择
    generateKuoHao_backtrack(left-1, right, track, res);
    track.pop_back();

    track.push_back(')');
    generateKuoHao_backtrack(left, right-1, track, res);
    track.pop_back();
}

vector<string> genKuoHao(int n){
    string track;
    vector<string> res;
    generateKuoHao_backtrack(n, n, track, res);
    return res;
}


// 3. 最长有效括号 (给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。)
// 常规动规解法： dp[i]表示以s[i]结尾的最长括号长度。以'('结尾的肯定是0，每两个字符检查一遍
//          case1 (): dp[i] = dp[i-2] + 2
//          case2 )): 如果i - dp[i-1] -1位置为'(', 那么 dp[i] = dp[i-1] + dp[i - dp[i-1] -1] + 2。否则dp[i] = 0（以i结尾没办法构成有效括号）
int longestValidParentheses_dp(string s){
    int maxlen = 0;
    int n = s.length();
    vector<int> dp(n, 0);

    for(int i = 1; i < n; i++){
        if(s[i] == ')'){
            if(s[i-1] == '('){
                if(i-2 < 0) {
                    dp[i] = 2;
                }
                else {
                    dp[i] = dp[i-2] + 2;
                }
            }
            else if(s[i-1] == ')' && i-dp[i-1]-1 >= 0 && s[i-dp[i-1]-1] == '('){
                if((i-dp[i-1]-2) < 0) {
                    dp[i] = dp[i - 1] + 2;
                }
                else{
                    dp[i] = dp[i-1] +  dp[i-dp[i-1]-2] + 2;
                }
            }
            maxlen = max(maxlen, dp[i]);
        }
    }
    return maxlen;
}


// 4. 有效括号的嵌套深度 - 分配两个字符串让深度最小
// 题目描述：给你一个「有效括号字符串」 seq，请你将其分成两个不相交的有效括号字符串，A 和 B，并使这两个字符串的深度最小。
//   划分方案用一个长度为 seq.length 的答案数组 answer 表示，编码规则如下：
//      answer[i] = 0，seq[i] 分给 A 。
//      answer[i] = 1，seq[i] 分给 B 。
//     输入：seq = "(()())"，  输出：[0,1,1,1,1,0]
// 思路：维护一个栈 s，从左至右遍历括号字符串中的每一个字符：
//      如果当前字符是 (，就把 ( 压入栈中，此时这个 ( 的嵌套深度为栈的高度；
//      如果当前字符是 )，此时这个 ) 的嵌套深度为栈的高度，随后再从栈中弹出一个 (。
//括号序列   ( ( ) ( ( ) ) ( ) )
//下标编号   0 1 2 3 4 5 6 7 8 9
//嵌套深度   1 2 2 2 3 3 2 2 2 1

//显然对半分配时最大值最小，要实现这样的对半分配，我们只需要把奇数层的 ( 分配给 A，偶数层的 ( 分配给 B 即可
//栈中只会存放 (，因此我们不需要维护一个真正的栈，只需要用一个变量模拟记录栈的大小。
vector<int> maxDepthAfterSplit(string seq) {
    int d = 0;
    vector<int> ans;
    for (char& c : seq){
        if (c == '(') {
            ++d;
            ans.push_back(d % 2);
        }
        else {
            ans.push_back(d % 2);
            --d;
        }
    }
    return ans;
}


void test_kuohao_match(){
    string s = "(){";
    assert(isValidKuoHao(s));
}

// 5. 删除无效的括号
// 题目描述：给你一个由若干括号和字母组成的字符串 s ，删除最小数量的无效括号，使得输入的字符串有效。
//      要求返回所有可能的结果。答案可以按 任意顺序 返回。
// 思路：由于题目要求我们将所有（最长）合法方案输出，因此不可能有别的优化，只能进行回溯「爆搜」
//基本思路：
//我们知道所有的合法方案，必然有左括号的数量与右括号数量相等。
//  首先我们令左括号的得分为 1；右括号的得分为 -1。那么对于合法的方案而言，必定满足最终得分为 0。
//  同时我们可以预处理出「爆搜」过程的最大得分： max = min(左括号的数量, 右括号的数量)
// PS.「爆搜」过程的最大得分必然是：合法左括号先全部出现在左边，之后使用最多的合法右括号进行匹配。
//枚举过程中出现字符分三种情况：
//普通字符：无须删除，直接添加
//左括号：如果当前得分不超过 max - 1 时，我们可以选择添加该左括号，也能选择不添加
//右括号：如果当前得分大于 0（说明有一个左括号可以与之匹配），我们可以选择添加该右括号，也能选择不添加
unordered_set<string> resset;
string res;
int maxscore, reslen;
void backtrack(const string& s, int score, string buf, int l, int r, int idx){
    if(l < 0 || r < 0 || score < 0 || score > maxscore) {
        return ;
    }
    if(l == 0 && r == 0 && buf.length() == reslen){
        resset.insert(buf);
    }
    if(idx == s.length()) return; // 需要在这里退出，因为上面不一定需要idx到最后
    if(s[idx] == '('){ // 做选择
        // 选择 添加左括号，+1分，继续往下
        backtrack(s, score+1, buf+'(', l, r, idx+1);
        // 选择 不添加左括号，相当于删除当前左括号，分数不变，继续往下
        backtrack(s, score, buf, l-1, r, idx+1);
    }else if(s[idx] == ')'){
        // 选择 添加右括号，-1分，继续往下
        backtrack(s, score-1, buf+')', l, r, idx+1);
        // 选择 不添加右括号，相当于删除当前右括号，分数不变，继续往下
        backtrack(s, score, buf, l, r-1, idx+1);
    }else{
        // 遇到普通字符，直接添加
        backtrack(s, score, buf+s[idx], l, r, idx+1);
    }
}
vector<string> removeInvalidParentheses(string s) {
    //假设“(”为+1分,")"为-1分，那么合规的字符串分数一定是0
    //分数一定不会是负数，因为那样意味着)比(多，不可能合规
    // maxpair就是所有可匹配的(都在左边，一直+1，能达到的最大分数
    int l = 0, r = 0;
    int ld = 0, rd = 0; // 统计需要删除的左右括号数量
    for(const auto& c : s){
        if(c == '(') {
            ld++;
            l++;
        }else if(c == ')') {
            if(ld != 0) {
                ld--; // 遇到可匹配的右括号 l-1
            }else {
                rd++; // 需要删除的右括号数量+1
            }
            r++;
        }
    }
    maxscore = min(l, r); // 能达到的最大分数
    reslen = s.length() - ld -rd; //删除需要删除的左括号和右括号后，字符串应该有的长度
    backtrack(s, 0, "", ld, rd, 0);
    return {resset.begin(), resset.end()};
}//时间复杂度（n*2^n）最坏情况下，每个位置都有两种选择
//空间O(n)


// 6. 翻转每队括号间的子串
string reverseParentheses(string s) {
    deque<char> de;
    for(char c: s) {
        if (c != ')') {
            de.push_back(c);
        } else {
            // 找到 ) 对应的子串word
            string word;
            while(de.back() != '(') {
                word.push_back(de.back());
                de.pop_back();
            }
            de.pop_back();
            // 将word 前插进入de
            copy(word.begin(), word.end(), back_inserter(de));
        }
    }

    return string(de.begin(), de.end());
}

#endif //DATASTRUCT_ALGORITHM_KUOHAO_MATCH_H

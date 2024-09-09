//
// Created by kai.chen on 2021/12/13.
//
//      1. 验证IP地址
//      2. 复原IP地址
//          2.1. IP地址与int整数的转换
//      3. 正则表达式匹配 .*
//      4. 通配符匹配 ?*
//
//   字符解码方法 见dp_1d
#ifndef DATASTRUCT_ALGORITHM_IP_ADDRESS_OP_H
#define DATASTRUCT_ALGORITHM_IP_ADDRESS_OP_H
#include <string>
using namespace std;

// 1. 验证IP地址
// 题目描述：
//      编写一个函数来验证输入的字符串是否是有效的 IPv4 或IPv6 地址。
//
//      如果是有效的 IPv4 地址，返回 "IPv4" ；如IP = "172.16.254.1"
//      如果是有效的 IPv6 地址，返回 "IPv6" ；如IP = "2001:0db8:85a3:0:0:8A2E:0370:7334"
//      如果不是上述类型的 IP 地址，返回 "Neither"；如IP = "256.256.256.256"
// 思路：
// (1)拆分后进行处理 (2)正则匹配
//   IPV4不合法情况 1>.分隔符过多 2>.分隔符挨着  3>.以0开头  4>.数过长  5>.数大于255  6>.分隔符间多余三个数
//   IPV6不合法情况 1>.分隔符过多 2>.分隔符挨着  3>.数过长   4.不在16进制范围  5>.分隔符间多余四个数
//
// 要求考虑问题的全面性
int count(const string& IP, char sep){
    int cnt = 0;
    for(char c: IP){
        if(c == sep) cnt++;
    }
    return cnt;
}
string validIPAddress(string queryIP) {
    int dots_count = count(queryIP, '.');
    int colon_count = count(queryIP, ':');

    if(dots_count == 3 && colon_count == 0){ //IPv4
        if(queryIP.back() == '.') return "Neither";
        queryIP += '.';
        string segment;
        for(char ch : queryIP){
            if(ch == '.'){
                // 分隔符挨着, 前导0
                if(segment.empty() || (segment.size() > 1 && segment[0] == '0')){
                    return  "Neither";
                }
                int val = stoi(segment);
                // 数字太大
                if(val < 0 || val > 255) return "Neither";
                segment.clear();
                continue;
            }
            // 不是数字
            if( !isdigit(ch) ) return "Neither";
            segment += ch;
            // 数字过长
            if(segment.size() > 3) return "Neither";
        }
        return "IPv4";
    }
    else if(dots_count == 0 && colon_count == 7){ // IPv6
        if(queryIP.back() == ':') return "Neither";
        queryIP += ':';
        string segment;
        for(char ch : queryIP){
            if(ch == ':'){
                if(segment.empty()) return "Neither";
                segment.clear();
                continue;
            }
            if(!isdigit(ch) && !(ch >= 'a' && ch <= 'f') && \
            !(ch >= 'A' && ch <= 'F')){
                return "Neither";
            }
            segment += ch;
            if(segment.size() > 4) return "Neither";
        }
        return "IPv6";
    }

    return "Neither";
}// 时间复杂度：O(n), 空间O(1)

void test_valid_ip(){
    string ip = "192.0.0.1";
    cout<< validIPAddress(ip)<< endl;
}


// 2. 复原IP地址
//  题目描述：一段数字，找出所有可能复原出的 IP 地址，如s = "25525511135"
//      输出：["255.255.11.135","255.255.111.35"]
//  思路：   回溯+判断合法性剪枝（需要注意不能有先导0）
vector<string> ans;
vector<int> segments;
// dfs(segStart, segId)表示我们正在从 s[segStart] 的位置开始，搜索 IP 地址中的第 segId 段，segID={0,1,2,3}
// 从segStart 开始，从小到大依次枚举当前这一段 IP 地址的结束位置 segEnd。
// 如果满足要求，就递归地进行下一段搜索，调用递归函数 dfs(segId+1,segEnd+1)。
// 在搜索的过程中，如果我们已经得到了全部的 4 段 IP 地址（即 segId=4），并且遍历完了整个字符串,  加入答案
void dfs(const string& s, int segId, int segStart) {
    // 如果找到了 4 段 IP 地址并且遍历完了字符串，那么就是一种答案
    if (segId == 4) {
        if (segStart == s.size()) {
            string ipAddr;
            for (int i = 0; i < 4; ++i) {
                ipAddr += to_string(segments[i]);
                if (i != 4 - 1) {
                    ipAddr += ".";
                }
            }
            ans.push_back(ipAddr);
        }
        return;
    }

    // 如果还没有找到 4 段 IP 地址就已经遍历完了字符串，那么提前回溯
    if (segStart == s.size()) {
        return;
    }

    // 由于不能有前导零，如果当前数字为 0，那么这一段 IP 地址只能为 0
    if (s[segStart] == '0') {
        segments[segId] = 0;
        dfs(s, segId + 1, segStart + 1);
    }

    // 一般情况，枚举每一种可能性并递归
    int addr = 0;
    for (int segEnd = segStart; segEnd < s.size(); ++segEnd) {
        addr = addr * 10 + (s[segEnd] - '0');
        if (addr > 0 && addr <= 0xFF) {  // (0,255]
            segments[segId] = addr;
            dfs(s, segId + 1, segEnd + 1);
        } else {
            break;
        }
    }
}
vector<string> restoreIpAddresses(string s) {
    segments.resize(4);
    dfs(s, 0, 0);
    return ans;
} // 时间复杂度：O(3^SEG_COUNT * |s|) 由于 IP 地址的每一段的位数不会超过 3，因此在递归的每一层，我们最多只会深入到下一层的 3 种情况。
// 由于 SEG_COUNT=4，对应着递归的最大层数，所以递归本身的时间复杂度为 O(3^SEG_COUNT)
// 如果我们复原出了一种满足题目要求的 IP 地址，那么需要 O(|s|)的时间将其加入答案数组中
// 空间复杂度：O(SEG_COUNT) 只计入除了用来存储答案数组以外的额外空间复杂度。递归使用的空间与递归的最大深度SEG_COUNT 成正比。


// 2.1 将ip地址转化为整数。
//  只需要将 第一段左移24位，第二段左移16位..最后或运算就行
unsigned ipToInt(string ip) {
    int l = ip.size();
    vector<int> ipList;
    //split
    for (int i = 0; i < l; i++) {
        int j = i;
        while (j < l && ip[j] != '.') j++;
        ipList.push_back(stoi(ip.substr(i, j - i)));
        i = j;
    }
    int n = ipList.size();
    unsigned res = 0;
    for (int i = 0; i < n; i++) {
        res = res << 8 | ipList[i];
    }
    return res;
}

string intToIp(unsigned num) {
    vector<string> ipList;
    string res = "";
    for(int i = 0; i < 4; i ++) {
        string seg = to_string(num & 255);
        ipList.push_back(seg);
        num = num >> 8;
    }
    reverse(ipList.begin(), ipList.end());
    for(int i = 0; i < 4; i ++) {
        if(i == 3) res += ipList[i];
        else res += ipList[i] + '.';
    }
    return res;
}

// 3. 正则表达式匹配
// 题目描述：.可以匹配一个任意字符，*可以让前面的字符重复任意次
//（不会出现*a, b** 这样的不合法模式串）
// 用下标i，j分别在两个字符串移动，最后移动到末尾就成功，否则失败
// 所以状态就是i,j两个位置，选择就是p[j]选择匹配几个字符
//      当s[i] == p[j]，dp()
bool dp(string& s, int i, string& p, int j){
    int m = s.size(), n = p.size();
    // base case
    if(j == n) return i == m; //case1模式串用完了，看文本串到了哪里
    if(i == m){ //case2文本串用完了，看模式串剩下的能否匹配空串
        if((n-j) % 2 == 1){ //能匹配空串必须是字符和*成对出现
            return false;
        }
        // 检查是否符合 a*b*c* 成对模式
        for(; j +1 < n; j += 2){
            if(p[j+1] != '*'){
                return false;
            }
        }
        return true;
    }

    string key = to_string(i) + "," + to_string(j);
    if(memo.count(key)) return memo[key]; // 消除重复计算

    bool res = false;
    if(s[i] == p[j] || p[j] == '.'){
        // j+1出现*通配符，可以匹配0次或多次   a, a*
        if(j < n-1 && p[j+1] == '*') {
            res = dp(s, i, p, j + 2) || dp(s, i + 1, p, j); // i往后移，j不动表示可以继续用a*匹配
        }
        // j+1没有出现*通配符，只能匹配1次
        else{
            res = dp(s, i+1, p, j+1);
        }
    }
    else{ // i,j位置发生不匹配
        // 检查j+1是否出现*通配符，有的话只能匹配0次，没有直接返回false
        if(j <n-1 && p[j+1] == '*'){
            res = dp(s, i, p, j+2);
        }
        else{
            res = false;
        }
    }
    memo[key] = res; // 备忘录
    return res;
} // 时间复杂度: O(m*n), m和n分别是s和p的长度
// 空间: O(m*n)


// 4. 通配符匹配
// 题目描述：？可以匹配一个任意字符，*可以匹配任意字符串包括""
// 用下标i，j分别在两个字符串移动，最后移动到末尾就成功，否则失败
//  模式p[j]，要么a~z,要么?,要么*
// dp定义：dp[i][j]表示s前i个字符和p的前j个字符是否匹配
bool isMatch(string s, string p) {
    int m = s.size();
    int n = p.size();
    vector<vector<bool>> dp(m + 1, vector<bool>(n + 1));
    // base case, dp[0][0]=0
    dp[0][0] = true;
    // base case, dp[i][0]=false, dp[0][j]当j前面都是*才可以匹配空字符
    for (int i = 1; i <= n; ++i) {
        if (p[i - 1] == '*') {
            dp[0][i] = true;
        }
        else {
            break;
        }
    }

    for (int i = 1; i <= m; ++i) { // i,j含义是第几个字符，从1开始
        for (int j = 1; j <= n; ++j) {
            if (p[j - 1] == '*') { // 第j的位置是*，对s的前i个字符没有任何要求
                // 选择：用这个*匹配空字符，还是用*匹配掉当前字符，并继续用*匹配前i-1个未匹配字符
                dp[i][j] = dp[i][j - 1] | dp[i - 1][j];
            }
            else if (s[i - 1] == p[j - 1]) { //
                dp[i][j] = dp[i - 1][j - 1];
            }
            else if(p[j - 1] == '?') {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }
    return dp[m][n];
} // 时间复杂度O(m*n), 空间复杂度O(m*n)



#endif //DATASTRUCT_ALGORITHM_IP_ADDRESS_OP_H

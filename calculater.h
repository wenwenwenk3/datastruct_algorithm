//
// Created by kai.chen on 2021/12/18.
//
//     1. 字符串转整数
//      1.1 字符串解码
//      1.2 分数到小数
//      1.3 整数转罗马数字
//      1.4 整数转十六进制数
//      1.5 整数反转
//     2. 实现计算器
//     3. 比较版本号
//

#ifndef DATASTRUCT_ALGORITHM_CALCULATER_H
#define DATASTRUCT_ALGORITHM_CALCULATER_H
#include <cctype>
#include <stack>
#include <algorithm>
using namespace std;

// 1. 字符串转整数
// 基本思路：for 1..n : x = x*10 + (s[i]-'0')
// 注意处理特殊情况：
//  包括：字符串为""，nullptr，0～9之外的字符，+号 -号
//   如果整数数超过 32 位有符号整数范围 [−2^31, 2^31− 1] ，需要截断这个整数，
//      使其保持在这个范围内。具体来说，小于 −2^31 的整数应该被固定为 −2^31 ，大于 2^31− 1 的整数应该被固定为 2^31− 1
//   注意这种情况 "words and 987"，输出0
int valid = 0;
int str2int(const string& s){
    int res = 0;
    int i = 0;
    bool isnegative = false;
    while (s[i] == ' '){ // 第一个判断，判断是否为空格
        i++;
    }
    if(s[i] == '-'){    // 第二个判断，判断是否为负数
        isnegative = true;  // 如果判断是负数，那么把isnegative赋值为-1。
    }
    if(s[i] == '+' || s[i]=='-'){  // 跳过首位字符+和——
        i++;
    }

    for(; i < s.size(); i++){
        if(isdigit(s[i])){
            int cur = (s[i] - '0');
            // 此处判断是否超过2的31次方和2的负31次方-1. 即INT_MAX 和 INT_MIN (-2147483648)
            if(res > INT_MAX / 10 || (res == INT_MAX / 10 && cur > 7)){
                return isnegative ? INT_MIN : INT_MAX;  // 判断正负，正则输出INT_MAX，负责输出INT_MIN
            }
            res = res*10 + cur;
        }
        else {
            valid = -1;
            break;
        }
    }
    return isnegative ? -res : res;
}

// 1.1 字符串解码
// 题目描述： 输入：s = "3[a]2[bc]"
//      输出："aaabcbc"
//  注意可能出现括号嵌套的情况，比如 2[a2[bc]]
// 思路：
//     字符分为：字母、数字和括号
//   当遇到数字时，解析数字并加入到栈
//   当遇到字母或左括号，直接进栈。
//   直到遇到右括号，开始出栈。出栈序列构成反转就是需要拼接的字符串。再取一个数字就是需要的次数。
// 具体可以使用 变长数组代替栈，方便的从栈底到栈顶遍历。
/*

func decodeString(s string) string {
	stk := []string{}
	for i := 0; i < len(s); {
		c := s[i]
		if c > '0' && c <= '9' { // 遇到数字，解析后加入栈
			l, r := i, i
			for r < len(s) && s[r] >= '0' && s[r] <= '9' {
				r++
			}
			stk = append(stk, s[l:r])
			i = r
		} else if c >= 'a' && c <= 'z' || c == '[' { // 遇到字符 或 [, 直接加入栈
			stk = append(stk, string(c))
			i++
		} else { // 遇到 ]
			substr := []string{}
			for len(stk) > 0 && stk[len(stk)-1] != "[" {
				substr = append(substr, stk[len(stk)-1])
				stk = stk[:len(stk)-1]
			}
			slices.Reverse(substr)
			stk = stk[:len(stk)-1]                     // 左括号出栈
			number, _ := strconv.Atoi(stk[len(stk)-1]) // 此时栈顶元素就是 数字 （需要重复的次数）
			stk = stk[:len(stk)-1]

			var strNeedJoin string
			for _, str := range substr {
				strNeedJoin += str
			}
			currStr := strings.Repeat(strNeedJoin, number)
			stk = append(stk, currStr)
			i++
		}
	}
	return strings.Join(stk, "")
}
 */
string decodeString(string s) {
    vector<string> stk;

    for(int i = 0; i < s.size(); ){
        if(s[i] >= '0' && s[i] <= '9'){
            // 遇到数字，解析数字 加入栈
            string numstr;
            while (isdigit(s[i])) {
                numstr.push_back(s[i]);
                i++;
            }
            stk.push_back(numstr);
        }
        else if(isalpha(s[i]) || s[i] == '['){
            // 遇到字母 或 [ 直接入栈
            stk.push_back(string(1, s[i]));
            i++;
        }
        else{ // 遇到 ]
            vector<string> substr;
            while(stk.back() != "["){
                substr.push_back(stk.back());
                stk.pop_back();
            }
            reverse(substr.begin(), substr.end());
            // 左括号出栈
            stk.pop_back();
            // 此时back就是需要重复的次数数字
            int count = stoi(stk.back());
            stk.pop_back();

            // 从substr中拼接需要的字符串
            string strNeedJoin;
            for (const string& str: substr) {
                strNeedJoin += str;
            }

            // 将 需要的字符串 * 需要重复的次数
            string strAfterRepeat;
            while(count--){
                strAfterRepeat += strNeedJoin;
            }

            stk.push_back(strAfterRepeat);
            i++;
        }
    }

    string res;
    for (const string& str: stk) {
        res += str;
    }
    return res;
}

// 1.2 分数到小数
// 题目描述： 输入：numerator = 1, denominator = 2 输出："0.5"
//     输入：numerator = 2, denominator = 3  输出："0.(6)"
// 思路：明确前提： 两个数相除要么是「有限位小数」，要么是「无限循环小数」，而不可能是「无限不循环小数」。
//   模拟一下除法的时候，一旦发现余数相同的情况，那么补零再除怎么都会出现循环小数。
string fractionToDecimal(int numerator, int denominator) {
    // 转 long 计算，防止溢出
    long a = numerator, b = denominator;
    // 如果本身能够整除，直接返回计算结果
    if (a % b == 0) {
        return to_string(a / b);
    }
    string res;
    // 如果其一为负数，先追加负号
    if(a * b < 0) {
        res.push_back('-');
    }
    a = abs(a), b = abs(b);
    // 计算整数部分
    res.append(to_string(a / b) + ".");
    // 余数部分
    long r = a % b;
    unordered_map<long, int> map;

    while (r != 0) {
        // 记录当前余数所在答案的位置，并继续模拟除法运算
        map[r] = res.length();
        r *= 10; // 补零
        res.append(to_string(r / b));
        r %= b;
        // 如果当前余数之前出现过，则将 [出现位置 到 当前位置] 的部分抠出来（循环小数部分）
        if(map.count(r)) {
            int p = map[r];
            int end = res.length();
            return res.substr(0, p) +"(" + res.substr(p, end)+ ")";
        }
    }
    return res;
}

// 1.3 整数转罗马数字
string intToRoman(int num) {
    string strs[] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
    int nums[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    string ans;
    for (int i = 0; num != 0; i++) {
        while (num >= nums[i]) {
            ans += strs[i];
            num -= nums[i];
        }
    }
    return ans;
}

// 1.4 整数转十六进制数
// 对于 0 到 9，数字本身就是十六进制数；
// 对于 10 到 15，将其转换为 a 到 f 中的对应字母。
string toHex(int num) {
    if (num == 0){
        return "0";
    }
    // 0-15映射成字符
    string num2char = "0123456789abcdef";
    // 转为为非符号数
    unsigned int n = num;
    string res = "";
    while (n > 0){
        res = num2char[n&15] + res;
        n >>= 4;  // n = n/16
    }
    return res;
}

// 1.5 整数反转
int reverse(int x) {
    int res = 0;
    while (x != 0) {
        if (res < INT_MIN / 10 || res > INT_MAX / 10) { // 注意溢出问题就行
            return 0;
        }
        int digit = x % 10;
        x /= 10;
        res = res * 10 + digit;
    }
    return res;
} // 时间复杂度: x 的十进制位数



// 2. 实现计算器
// 题目描述
// 思路，拆分复杂问题：第一步处理字符串转整数
//      第二步处理加减法，将"1-12+3"看成两两一对，存到栈中，求和就是
//      第三步处理乘除法，和处理加减法一样，把字符串分解成符号和数字的组合就是
//            需要注意乘除法优先级高于加减法，所以考虑压栈的时候直接和栈顶数字结合在放入栈
//            需要注意处理空格字符，判断数字的时候加上条件跳过空格就是
//      第四步处理括号，最难点：
//          思路就是，无论括号多少层，其实就是递归搞定
//           如 3*(4 - 5/2) - 6 = 3*cal(4 - 5/2) -6
//           遇到（ 开始递归，遇到）结束递归
int calculate_helper(list<char> &s) {
    stack<int> stk;
    char sign = '+';
    int num = 0;

    while(!s.empty()){
        char c = s.front(); s.pop_front();
        if(isdigit(c)){
            num = 10* num + (c - '0');
        }
        if(c == '('){
            num = calculate_helper(s);
        }
        int tmp;
        if((!isdigit(c) && c != ' ') || s.empty()) { // 符号或最后一个数字
            switch (sign) {
                case '+':
                    stk.push(num);
                    break;
                case '-':
                    stk.push(-num);
                    break;
                case '*':
                    tmp = stk.top(); stk.pop();
                    stk.push(tmp * num);
                    break;
                case '/':
                    tmp = stk.top(); stk.pop();
                    stk.push(tmp / num);
                    break;
            }
            num = 0;
            sign = c;
        }
        if(c == ')') break;
    }
    int sum_stk = 0;
    while(!stk.empty()){
        sum_stk += stk.top();
        stk.pop();
    }
    return sum_stk;
}
int calculate(string s) {
    list<char> li;
    for(int i=0; i <s.size();i++){
        li.push_back(s[i]);
    }
    return calculate_helper(li);
}

void test_calculate(){
    // cout<< calculate("1+1")<<endl;
    cout<< calculate("(4+5+2)")<<endl;
    cout<< calculate("(4+5+2)-3)")<<endl;
    // cout<< calculate("(1+(4+5+2)-3)+(6+8)")<<endl;
}

// 3. 比较版本号
// 题目描述：
//  输入：version1 = "1.01", version2 = "1.001"
//  输出：0   解释：忽略前导零，"01" 和 "001" 都表示相同的整数 "1"
// 思路：(1) 字符串转整数 再比较
//      (2) 原地比较
int compareVersion(string version1, string version2) {
    int n = version1.size(), m = version2.size();
    int i = 0, j = 0;
    while (i < n || j < m) {
        int x = 0;
        for (; i < n && version1[i] != '.'; ++i) {
            x = x * 10 + version1[i] - '0';
        }
        ++i; // 跳过点号
        int y = 0;
        for (; j < m && version2[j] != '.'; ++j) {
            y = y * 10 + version2[j] - '0';
        }
        ++j; // 跳过点号
        if (x != y) {
            return x > y ? 1 : -1;
        }
    }
    return 0;
}


#endif //DATASTRUCT_ALGORITHM_CALCULATER_H

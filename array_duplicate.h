//
// Created by kai.chen on 2021/12/17.
//  数组去重：快慢指针技巧
//
//      1. 找出数组中的唯一重复数据
//      2. 找出数组中的所有重复数据 1 ≤ a[i] ≤ n
//        2.1 找出只出现一次的数，其他都出现k次 见bit_op.h

//      3. 删除排序数组中的重复数据
//          3.延伸 删除排序数组中的重复数据II（使每个元素 最多出现两次/k次）
//          3.1 删除排序链表的重复数据
//          3.2 删除排序链表的重复数据II (延伸到重复数字都要被删掉)
//
//      4. 删除字符串中的相邻重复项
//        4.1变体 空格替换
//        附4.2 整数替换
//        4.3. 移动零到末尾并保持其他元素顺序
//        4.4 字符串去重重复字母 （或叫不同字符的最小子序列）
//        4.5 复写0

//      5. 错误的集合 (寻找重复缺失数字)
//      6. 缺失的第一个正数
//
//      7. 1～n 整数中 1 出现的次数
#ifndef DATASTRUCT_ALGORITHM_ARRAY_DUPLICATE_H
#define DATASTRUCT_ALGORITHM_ARRAY_DUPLICATE_H
#include <vector>
#include "list_op.h"
using namespace std;

// 1. 找出数组中的唯一重复数据
// 题目描述：给定一个包含n + 1 个整数的数组nums，其数字都在 1 到 n之间（包括 1 和 n），可知至少存在一个重复的整数。
//      假设 nums 只有 一个重复的整数 ，找出 这个重复的数 。
// 思路：
//     floy判圈法：类似链表环快慢指针
//      假设每个位置 i连一条 i→nums[i] 的边。由于存在的重复的数字target，因此 target 这个位置一定有起码两条指向它的边，因此整张图一定存在环。找的是环的入口
//      设环的周长为L，当 fast 追上了 slow 时，假设fast多走了 k圈，fast走了b步，slow走了a步
int findDuplicate(vector<int>& nums) {
    int slow = nums[0], fast = nums[nums[0]];
    while (slow != fast){
        slow = nums[slow]; // slow每次走一条边
        fast = nums[nums[fast]]; // fast每次走两条边
    }
    // fast 追上了 slow

    slow = 0;
    while (slow != fast) {
        slow = nums[slow];
        fast = nums[fast];
    }
    return slow;
}


// 2. 找出数组中的重复数据 1 ≤ a[i] ≤ n
// 题目描述：给定一个整数数组 a，其中1 ≤ a[i] ≤ n （n为数组长度）, 其中有些元素出现两次而其他元素出现一次。
//      找到所有出现两次的元素
// 思路：
//    注意到1<=nums[i]<=n(数组长度)，所以（nums[i]-1）可以成为nums中的下标，
//      记idx=nums[i]-1
//    所以: 遍历，每次扫描到nums[i]每出现过一次之后对nums[idx] += n,(其中idx=nums[i]-1)
//      这样可以确保当nums[idx] > 2*n时 就能表示 nums[i](即index+1) 出现过两次。
vector<int> findDuplicates2(vector<int>& nums){
    vector<int> res;
    int n=nums.size();
    for(int i = 0; i < nums.size(); i++){
        int idx = (nums[i] - 1) % n;
        nums[idx] += n;

        if(nums[idx] > 2 * n){ //当nums[idx] > 2*n时 就能表示 nums[i](即index+1) 出现过两次。
            res.push_back(idx+1);
        }
    }
    return res;
}
// 思路：原地哈希
// 数组nums下标范围为[0, nums.size() - 1]；
// 数组内元素范围为[1, max(num)]；
// 遍历一遍nums，每个元素都可以映射为nums数组中的唯一下标，如果任意元素出现过两次，那么该下标会被访问两次
//        在第一次访问此下标时，对其进行标记(比如对其取反)，当第二次访问到该位置时，若发现已经被标记过，说明当前元素已经是第二次出现了。
// 加入答案。
vector<int> findDuplicates(vector<int>& nums) {
    vector<int> res;
    for(const int& num : nums){
        if(nums[abs(num) - 1] < 0) res.emplace_back(abs(num));
        else nums[abs(num) - 1] *= -1;
    }
    return res;
}




// 3. 删除排序数组中的重复数据
//  思路：fast走在前面，slow走在后面，每次fast和slow不相等时，将fast赋值给slow
int removeDuplicates(vector<int>& nums){
    int n = nums.size();
    if(n == 0) return 0;
    int slow = 0, fast = 1;
    while(fast < n){
        if(nums[fast] != nums[slow]){
            slow++;
            nums[slow] = nums[fast];
        }
        fast++;
    }
    return slow+1; // 长度为 最大索引+1
}

// 3. 延伸 删除排序数组中的重复数据II（使每个元素 最多出现两次/k次）
// 思路：因为给定数组是有序的，所以相同元素必然连续
//  由于是保留 k 个相同数字，对于前 k 个数字，我们可以直接保留
//  对于后面的任意数字，能够保留的前提是：与当前写入的位置前面的第 k 个元素进行比较，不相同则保留
int removehelper(vector<int>& nums, int k) {
    int len = 0;
    for(auto num : nums){
        if(len < k || nums[len-k] != num){
            nums[len++] = num;
        }
    }
    return len;
}
int removeDuplicatesII(vector<int>& nums) {
    return removehelper(nums, 2);
}



// 3.1 删除排序链表的重复数据
ListNode* deleteDuplicates_(ListNode* head){ // 延伸到排序链表
    if(head == nullptr || head->next == nullptr) return head;
    ListNode* slow = head, *fast = head->next;
    while(fast != nullptr){
        if(fast->val != slow->val){
            slow->next = fast;
            slow = slow->next;
        }
        fast = fast->next;
    }
    slow->next = nullptr; // 断开后面的连接
    return head;
}

// 3.2 删除排序链表的重复数据II (延伸到重复数字都要被删掉)
//
ListNode* deleteDuplicatesII(ListNode* head){ // [1,2,3,3,4,4,5] -> [1,2,5]
    if(head == nullptr || head->next == nullptr) return head;

    // 由于这次头节点可能被删掉，新增加一个dummy节点
    ListNode* dummy = new ListNode(0, head);
    ListNode* prev = dummy, *left = head, *right = head->next;
    // 每次考虑的元素其实是 left
    while(right != nullptr){
        if(right->val == left->val){
            // right往后走直到找出和第一个和左边不相等的点
            while(right && left->val == right->val){
                right = right->next;
            }
            prev->next = right;
            left = right;
            if(right != nullptr){
                right = right->next;
            }
        }
        else{
            prev = left;
            left = right;
            right = right->next;
        }
    }
    return dummy->next;
}


// 4. 删除字符串中的相邻重复项
// 需要注意删除重复项之后可能有新的重复项产生
//如输入："abbaca"， 输出："ca"
// 思路：很明显想到了栈这种结构，和栈顶相等抹掉就是。这其实和括号匹配非常类似
string removeDuplicates(string s) {
    string stk;
    for (char ch : s) {
        if (!stk.empty() && stk.back() == ch) {
            stk.pop_back();
        } else {
            stk.push_back(ch);
        }
    }
    return stk;
}

// 4.1变体 空格替换
string replaceSpace(string s) {
    int count = 0, n = s.size();
    // 统计空格数量
    for (char c : s) {
        if (c == ' ') count++;
    }
    // 修改 s 长度
    s.resize(n + 2 * count);
    // 倒序遍历修改
    for(int i = n - 1, j = n - 1; i < j; i--, j--) {
        if (s[i] != ' '){
            s[j] = s[i];
        }
        else { // 把空格替换成 %20
            s[j] = '0';
            s[j - 1] = '2';
            s[j - 2] = '%';
            j -= 2;
        }
    }
    return s;
}

// 附4.2 整数替换
// 题目描述：给定一个正整数n ，你可以做如下操作：
//      如果n是偶数，则用n / 2替换n 。
//      如果n是奇数，则可以用n + 1或n - 1替换n 。
//    求n变为 1 所需的最小替换次数是多少？
//
// 思路：回溯，或者叫dfs
unordered_map<int, int> map;
int dfs(int n) {
    if (n == 1) return 0; // base case
    if (map.count(n)) return map[n]; // 备忘录剪枝

    int steps;
    // 做选择
    if(n % 2 == 0){
        steps = dfs(n / 2) + 1;
    } else {
        steps = min(dfs(n + 1), dfs(n - 1)) + 1;
    }

    map[n] = steps; // 加入备忘录

    return steps;
} // 时间复杂度O(logN), 空间复杂度为O(logN) 递归栈的需要
// 整数n特别大时 注意n+1可能会越界, 可以用 steps = min(dfs(n / 2), dfs(n / 2 + 1)) + 2 解决，因为奇数加一减一后肯定为偶数
int integerReplacement(int n) {
    return dfs(n);
}

// 4.3 移动零到末尾并保持其他元素顺序
//   输入：[0,1,0,3,12]   输出：[1,3,12,0,0]
// 思路： 双指针：从前往后跑，
//      左指针指向当前已经处理好的序列的尾部，左指针左边均为非零数，
//      右指针指向待处理序列的头部。 左右指针中间均为零。
// 都是将左指针的零与右指针的非零数交换
void moveZeroes(vector<int>& nums) {
    int n = nums.size(), left = 0, right = 0;
    while (right < n) {
        if (nums[right] != 0) { // 每次右指针指向非零数，则将左右指针对应的数交换
            if(left != right) swap(nums[left], nums[right]);
            left++;
        }
        right++;
    }
}

// 4.4 字符串去重重复字母 （或叫不同字符的最小子序列）
//题目描述
// 要求一、要去重。
//
//要求二、去重字符串中的字符顺序不能打乱 s 中字符出现的相对顺序。
//
//要求三、在所有符合上一条要求的去重字符串中，字典序最小的作为最终结果。
// https://leetcode-cn.com/problems/remove-duplicate-letters/solution/you-qian-ru-shen-dan-diao-zhan-si-lu-qu-chu-zhong-/
string removeDuplicateLetters(string s) {
    stack<char> stk;
    bool instack[256];
    memset(instack, 0, sizeof(instack));

    vector<int> count(256, 0);
    for (int i = 0; i < s.length(); i++) {
        count[s[i]-'0']++;
    }

    for(const auto& c : s){
        count[c-'0']--;
        if(instack[c-'0'] == false){
            while(!stk.empty() && c < stk.top()){
                if(count[stk.top()-'0'] > 0) {
                    instack[stk.top()-'0'] = false;
                    stk.pop();
                } else break;
            }
            stk.push(c);
            instack[c-'0'] = true;
        }
    }
    string res;
    while(!stk.empty()){
        res += stk.top();
        stk.pop();
    }

    reverse(res.begin(), res.end());
    return res;
}

// 4.5 复写0
// 题目描述：将数组每个零都复写一遍，并将其余的元素向右平移。要求就地修改
//   例如，输入：[1,0,2,3,0,4,5,0] 修改后：[1,0,0,2,3,0,0,4]
// 思路：可以从后往前填
void duplicateZeros(vector<int>& arr){
    int n = arr.size();
    int i = 0, j = 0;
    while(j < n){ // 当 j 走到的位置已超出数组长度，此时 i 也停在了该被截断的位置的下一位
        if(arr[i] == 0) j++;
        i++,j++;
    }
    i--,j--; // 先将 i 和 j 同时往回走一步
    while(i >= 0){
        if(j < n) {
            arr[j] = arr[i];
        }
        if(arr[i] == 0 && --j >= 0) {
            arr[j] = 0;
        }
        i--,j--;
    }
}


// 5. 错误的集合 (寻找缺失重复数字)
// 题目描述：本来装着 [1..N] 这 N 个元素
//  但因为某些数发生了重复，需要找到 nums 中的重复元素和缺失元素的值
//  例如：nums = [1,2,2,4]，算法返回 [2,3]
//
//  常规思路：1) 哈希表 统计出现的次数
//           2) 排序 寻找重复的数字较为简单，如果相邻的两个元素相等，则该元素为重复的数字。
///                 寻找丢失的数字相对复杂，如果丢失的数字大于 1 且小于 n，两个元素的差等于 2 之间的值就是缺失元素
///                        如果，丢失的数字是 1 或 n，另外判断
//  进阶思路：
//   暂且将 nums 中的元素变为 [0..N-1]，这样每个元素就和一个数组索引完全对应了，这样方便理解一些。
//   找到这个重复对应的索引，不就是找到了那个重复元素么？找到那个没有元素对应的索引，不就是找到了那个缺失的元素了么？

// 精妙之处： 通过将每个索引对应的元素变成负数，以表示这个索引被对应过一次了：
vector<int> findErrorNums(vector<int>& nums) {
    int n = nums.size();
    int dup = -1;
    for (int i = 0; i < n; i++) {
        // 现在的元素是从 1 开始的
        int index = abs(nums[i]) - 1;
        // nums[index] 小于 0 则说明重复访问
        if (nums[index] < 0)
            dup = abs(nums[i]);
        else
            nums[index] *= -1;
    }

    int missing = -1;
    for (int i = 0; i < n; i++) {
        // nums[i] 大于 0 则说明该索引没有访问
        if (nums[i] > 0) {
            // 将索引转换成元素
            missing = i + 1;
        }
    }
    return {dup, missing};
}

vector<int> findErrorNums_by_hash(vector<int>& nums) {
    vector<int> errorNums(2);
    int n = nums.size();
    unordered_map<int, int> mp;
    for (auto& num : nums) {
        mp[num]++;
    }
    for (int i = 1; i <= n; i++) {
        int count = mp[i];
        if (count == 2) {
            errorNums[0] = i;
        } else if (count == 0) {
            errorNums[1] = i;
        }
    }
    return errorNums;
}

// 6. 第一个缺失的正数
//  题目描述： 未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数，要求时间复杂度O(N), 空间复杂度O(1)
//  示例 1：
//输入：nums = [1,2,0]
//输出：3
//  示例 2：
//输入：nums = [3,4,-1,1]
//输出：2
//
// 思考：
// (1) 用数组所有的数建立哈希表，随后从 1 开始依次枚举正整数，并判断其是否在哈希表中；时间O(N)，空间O(N).
// (2) 从 1 开始依次枚举正整数，并遍历数组，判断其是否在数组中；时间O(N^2),空间O(1)
//
// 基于方法1 原地哈希：缺失的数一定在[1,N]内，如果不在的话那么数字(1,N)都在，最小缺失正数是N+1

int firstMissingPositive(vector<int>& nums) { // 举例：[3,4,-1,1,9,-5]，n=6
    int n = nums.size();
    for (int& num: nums) {
        if (num <= 0) num = n + 1; // 所有小于等于 0 的数修改为 N+1 => [3,4,7,1,9,7]
    }

    for (int i = 0; i < n; ++i) {
        int num = abs(nums[i]);
        if (num <= n) {
            nums[num - 1] = -abs(nums[num - 1]); // 把<=6的元素 "对应第x个位置" 变为负数 => [-3,4,-7,-1,9,7]
        }
    }
    for (int i = 0; i < n; ++i) {
        if (nums[i] > 0) { // 找到第一个 大于0 的数，的元素下标
            return i + 1;
        }
    } // 在遍历完成之后，如果数组中的每一个数都是负数，那么答案是 N+1，否则答案是第一个正数的位置加 1。
    return n + 1;
}

// 7. 1～n 整数中 1 出现的次数
// 题目描述：输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。
//  例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。
// 总结题：
//   可以简化为“固定某个位为1，求其他位的组合数”。比如求20213这个数，那么按照下面的步骤进行求解：
//固定个位数为1，个位数原本的数大于1，那么可以直接置为1（如果小于1，需要向高位借1），那么个位数为1的组合数就是2022，也就是高位从0000到2021;
//固定十位数为1，十位数原本就是1，十位数不用置位，因此高位可以从000-201，低位从0-9，加上高位为202，低位从0-3，也就是组合数为202*10+4=2024；
//固定百位数为1，百位数大于1，那么直接置1，因而高位可以为00-20，低位0-99，即组合数为21*100=2100;
//固定千位数为1，千位数为0小于1，向前借1并置该位为1，那么此时高位只剩下1，因此高位可以为0-1，低位000-999共2000;
//固定万位数为1，万位大于1，置为1，减小了该位，低位就可以从0-9，所以高位为0，低位从0000-9999，所以组合数为10000;
//综上，20213中出现1的次数为2022+2024+2100+2000+10000=18146.
// https://leetcode.cn/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/solution/by-lao-hang-n-6umn/
int countDigitOne(int n) {
    if(n == 0)  return 0;
    int base = 1;
    long long ans = 0;
    while(1ll * n >= pow(10, base-1)) {
        int high = n / pow(10, base);               // 计算高位
        int low = n % (int)pow(10, base-1);         // 计算低位
        int cur = n - high * pow(10, base) - low;
        cur /= pow(10, base-1);                     // 计算当前位
        // 判断属于哪种情况
        if(cur > 1)
            ans += (high+1)*pow(10, base-1);
        else if(cur == 1)
            ans += high*pow(10, base-1) + low + 1;
        else
            ans += high*pow(10, base-1);
        ++base;                                     // 进行下一位处理
    }
    return (int)ans;
} // 时间: O(logN), n 包含的数位个数与 n 呈对数关系。
// 空间: O(1)


#endif //DATASTRUCT_ALGORITHM_ARRAY_DUPLICATE_H

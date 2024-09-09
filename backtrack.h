//
// Created by kai.chen on 2021/12/4.
//
// topic.回溯问题：
// 排列组合子集问题MS题库涉及数量: 4+3+2 = 9道
// 1. 排列，
//      1.1 全排列II(带重复数字)
//      1.2 下一个排列
//      1.3 第k个排列
//      1.4 数组元素组成的最大数
//      1.5 按字典序返回范围 [1, n] 内所有整数
// 2. 组合
// 3. 子集
//     2.1 组合总和
//     2.2 组合总和II
//     2.3 电话号码的字母组合
// // 3.1 划分为k个相等的子集

//  4. 形成3的最大倍数：
//
//  5. 可能组成目标和的表达式数目
//  6. 找所有递增子序列
//
#ifndef DATASTRUCT_ALGORITHM_BACKTRACK_H
#define DATASTRUCT_ALGORITHM_BACKTRACK_H
#include <vector>
#include <algorithm>
using namespace std;

// 1. 排列
vector<vector<int>> result;
void backtrack_pailie_by_swap(vector<int> &nums, int l){
    if(l == nums.size()){
        result.push_back(nums);
        return;
    }
    int sz = nums.size();
    for(int i = l; i < sz; i++){
        swap(nums[i],nums[l]);
        backtrack_pailie_by_swap(nums, l+1);
        swap(nums[l],nums[i]);
    }
}
vector<vector<int>> permute_swap(vector<int> &nums){// 全排列交换法
    backtrack_pailie_by_swap(nums, 0);
    return result;
} // 时间复杂度：O(n×n!)

//vector<vector<int>> result;
void backtrack_pailie_by_normal(vector<int> &nums, vector<int>& track){
    if(track.size() == nums.size()){
        result.push_back(track);
        return;
    }
    for(int i = 0; i < nums.size(); i++){
        auto iter = find(track.begin(), track.end(), nums[i]);
        // 排除不合法的选择
        if(iter != track.end()) continue;
        // 做选择
        track.push_back(nums[i]);
        backtrack(nums, track);
        // 取消选择
        track.pop_back();
    }
}
vector<vector<int>> permute_normal(vector<int> &nums){ // 全排列常规法
    vector<int> track;
    backtrack_pailie_by_normal(nums, track);
    return result;
}

// 变体1.1 全排列II(带重复数字)
// 题目描述：给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
//  输入：nums = [1,1,2]
//输出：
//[[1,1,2],
// [1,2,1],
// [2,1,1]]
// 思路： 假设成有 n 个排列成一行的空格，我们需要从左往右依次填入题目给定的 n 个数，每个数只能使用一次
//     如何保证 重复数字只会被填入一次
//      通过对原数组排序，保证相同的数字都相邻，然后每次填入的数一定是这个数所在重复数集合中「从左往右第一个未被填过的数字」
vector<vector<int>> result;
vector<bool> memo;
void backtrack_pailieII(vector<int> &nums, int idx, vector<int>& track){
    if(idx == nums.size()){
        result.push_back(track);
        return;
    }
    int sz = nums.size();
    for(int i = 0; i < sz; i++){
        if (memo[i] || (i >= 1 && nums[i] == nums[i - 1] && memo[i - 1])) {
            // 若nums[i]已经使用了 || ( nums[i]和nums[i-1]相等 且 nums[i-1]使用 )
            continue;
        }

        track.push_back(nums[i]);
        memo[i] = true;
        backtrack_pailieII(nums, idx+1, track);
        memo[i] = false;
        track.pop_back();
    }
}
vector<vector<int>> permuteII(vector<int> &nums){// 全排列带重复数字
    vector<int> track;
    memo.resize(nums.size());
    sort(nums.begin(), nums.end());
    backtrack_pailieII(nums, 0, track);
    return result;
}

// 1.2 下一个排列
//题目描述：算法需要将给定数字序列重新排列成字典序中下一个更大的排列（即，组合出下一个更大的整数）。
//  如果不存在下一个更大的排列，则将数字重新排列成最小的排列
// 输入：nums = [1,2,3] 输出：[1,3,2]
// 思路：就是找更大的数，且增大的幅度尽可能小。理解一下数字升序即为最小，数字降序即为最大
// 具体步骤：首先从后往前找第一个升序对 a[i]<a[i+1]。这样「较小数」为a[i]。此时[i+1,n) 必然是下降序列。
//      若找到位置i, 从[i+1,n)中从后往前找出第一个j 让a[j]>a[i]的数，即为「较大数」
//      交换a[i] a[j], 此时[i+1,n)必为降序。可以直接反转区间 [i+1,n) 使其变为升序。
// 如 【1，2，5，4，3】 a[i]=2, a[j]=3, 交换后[1,3,5,4,2], 翻转区间[5,4,2]后 => 得到结果【1,3,2,4,5】
void nextPermutation(vector<int>& nums) {
    int i = nums.size() - 2;
    // 从后往前找第一个升序对 a[i]<a[i+1]
    for(;i >= 0 && nums[i] >= nums[i + 1];i--);
    // 若找到位置i, 从[i+1,n)中
    if (i >= 0) {
        int j = nums.size() - 1;
        // 从[i+1,n)中从后往前找出第一个j 让a[j]>a[i]的数
        for(;j >= 0 && nums[i] >= nums[j];j--);
        swap(nums[i], nums[j]);
    }
    // 如果没有找到升序对位置i,此时i=-1,那也是直接翻转成升序即为最小
    reverse(nums.begin()+i+1, nums.end());
}// 时间：O(N)

// 1.3 第k个排列
// 题目描述：给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。按大小顺序列出所有排列情况：
//      给定整数n，返回集合[1..n]的第k个排列
//  1: "123"
//  2: "132"
//  3: "213"
// 思路：这题hard答案纯粹是个数学题，不过可以用笨方法拆分问题：全排列，第k大
// 或者用 下一个排列的思路，从最小的开始 下k个排列 STL next_permutation
string getPermutation(int n, int k) {
    string str;
    char i = '0';
    while(n--){
        i++;
        str += i;
    }
    while(--k){
        next_permutation(str.begin(), str.end());
    }
    return str;
} // 下一个排列next_permutation时间复杂度为O(n)
// 自己实现下一个排列
bool nextPermutation(string& s) {
    int i = s.size() - 2;
    while (i >= 0 && s[i] >= s[i + 1]) {
        i--;
    }
    if (i < 0) {
        return false;
    }
    int j = s.size() - 1;
    while (j >= 0 && s[i] >= s[j]) {
        j--;
    }
    swap(s[i], s[j]);
    reverse(s.begin() + i + 1, s.end());
    return true;
}

// 1.4 数组元素组成的最大数
// 首先想到的就是 较大的数放在前面就能让数字最大。但处理不了有相同数字开头的情况， 如[2, 21], [2,23]
//  对于相同数字开头的情况，其实按照字典序排就好了
string largestNumber(vector<int> &nums){
    // 转成string
    vector<string> numstr;
    for(auto num : nums) {
        numstr.push_back(to_string(num));
    }
    // 按字典序 从大到小 排序
    sort(numstr.begin(), numstr.end(), [](const string &x, const string &y){
        return x + y > y + x;
    });

    // 拼接字符串
    string res;
    for(const string &str : numstr) {
        res += str;
    }
    if (res[0] == '0') { // 特殊情况最大值是0，容易漏掉考虑
        return "0";
    }
    return res;
}

// 1.5 按字典序返回范围 [1, n] 内所有整数
vector<int> lexicalOrder(int n) {
    // vector
    vector<int> ret(n);
    int nextNum = 1;
    for (int i = 0; i < n; i++) {
        ret[i] = nextNum;
        if (nextNum * 10 <= n) {
            nextNum *= 10;
        } else {
            //
            while (nextNum % 10 == 9 || nextNum + 1 > n) {
                nextNum /= 10;
            }
            nextNum++;
        }
    }
    return ret;
} // 时间复杂度 O(N), 空间 O(1)

// 2. 组合
// 题目描述：找出从[1, n]中k个数字的所有组合
int n, k;
void backtrack_zuhe(int start, vector<int>& track){
    if(track.size() == k){
        res.push_back(track);
        return;
    }
    for(int i = start; i <= n; i++){
        track.push_back(i);
        backtrack_zuhe(i+1, track);
        track.pop_back();
    }
}
vector<vector<int>> combine(int N, int K) {
    vector<int> track;
    n = N;
    k = K;
    backtrack(1, track);
    return res;
}

// 3. 子集
void backtrack_ziji(vector<int>& nums, int start, vector<int>& track){
    res.push_back(track);
    for(int i = start; i < nums.size(); i++){
        track.push_back(nums[i]);
        backtrack_ziji(nums, i+1, track);
        track.pop_back();
    }
}
vector<vector<int>> subsets(vector<int>& nums) {
    vector<int> track;
    backtrack_ziji(nums, 0, track);
    return res;
}



// 2.1 组合总和
// 题目描述：找出 nums 中所有可以使数字和为目标数 target 的唯一组合。
// （candidates 中的数字可以无限制重复被选取。如果至少一个所选数字数量不同，则两种组合是唯一的。）
//   思路：回溯start和target，表示当前递归到了start位置，还需要选择和为target的数字放入track
vector<int> nums;
vector<vector<int>> res;
void backtrack_zuhe_sum(int start, int target, vector<int>& track){
    if(start >= nums.size()){
        return;
    }
    if(target == 0){
        res.push_back(track);
        return;
    }

    // 跳过当前数nums[start]
    backtrack_zuhe_sum(start+1, target, track);
    // 选择当前数nums[start]
    if(target - nums[start] >= 0){
        track.push_back(nums[start]);
        backtrack_zuhe_sum(start, target - nums[start], track); // 注意这里start可以被重复选取
        track.pop_back();
    }
}
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<int> track;
    nums = candidates;
    backtrack_zuhe_sum(0, target, track);
    return res;
}
// 时间复杂度：O(S)，其中 S 为所有可行解的长度之和。上界是O(n*2^n), n 个位置每次考虑选或者不选. 有>=0的剪枝不会达到。
// 空间复杂度：O(target)。除答案数组外，空间复杂度取决于递归的栈深度，在最差情况下需要递归 O(target) 层。


// 2.2 组合总和II
// 找出 candidates 中所有可以使数字和为 target 的组合。
// （candidates 中的每个数字在每个组合中只能使用一次。）
// 难点在 基于组合总和的去重上：
//      思路是：(1).可以使用 哈希表  (2).可以首先需要对数组排序，然后按顺序搜索再递归的时候判断此时节点上的数与上一个数是否相同，如果是则跳过。
void DFS(int start, int target, vector<int>& track) {
    if (target == 0) {
        res.push_back(track);
        return;
    }

    for (int i = start; i < candidates.size() && target - candidates[i] >= 0; i++) {
        if (i > start && candidates[i] == candidates[i - 1])
            continue;
        track.push_back(candidates[i]);
        // 元素不可重复利用，使用下一个即i+1
        DFS(i + 1, target - candidates[i], track);
        track.pop_back();
    }
}
vector<vector<int>> combinationSum2(vector<int> &candidates, int target) {
    sort(candidates.begin(), candidates.end());
    this->candidates = candidates;
    vector<int> track;
    DFS(0, target, track);
    return res;
} // 时间复杂度: O(n * 2^n) 每个数字可以考虑选/不选，并且每得到一个满足条件的组合需要O(n)时间push_back
// 空间: O(n) 递归栈深度和track都不需要n空间

// 2.3 电话号码的字母总和
// 题目描述：给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
// 思路：首先使用哈希表存储每个数字对应的所有可能的字母，然后进行回溯操作。
//      每次取电话号码的一位数字，从哈希表中获得该数字对应的所有可能的字母，
//      并将其中的一个字母插入到已有的字母排列后面，然后继续处理电话号码的后一位数字，
//      直到处理完电话号码中的所有数字，即得到一个完整的字母排列
unordered_map<char, string> phoneMap;
vector<string> res;
void backtrack(const string& digits, int index, string& track) {
    if (index == digits.length()) {
        res.push_back(track);
    } else {
        char digit = digits[index];
        const string& letters = phoneMap.at(digit);
        for (const char& letter: letters) {
            track.push_back(letter);
            backtrack(digits, index + 1, track);
            track.pop_back();
        }
    }
}
vector<string> letterCombinations(string digits) {
    vector<string> combinations;
    if (digits.empty()) {
        return combinations;
    }
    phoneMap = {
            {'2', "abc"},
            {'3', "def"},
            {'4', "ghi"},
            {'5', "jkl"},
            {'6', "mno"},
            {'7', "pqrs"},
            {'8', "tuv"},
            {'9', "wxyz"}
    };
    string track;
    backtrack(digits, 0, track);
    return res;
}// 思考T9键盘问题，不就是再判断一下是否生成的字符串符合有效单词，可以使用map判断
//    vector<string> getValidT9Words(string num, vector<string>& words) {


// 3.1 划分为k个相等的子集
// 题目描述：找出是否有可能把这个数组分成 k 个非空子集，其总和都相等。
// 思路：   1、遍历nums中所有数字，决定哪些数字需要放到当前集合中；
//         2、如果当前集合装满了（集合内数字和达到target），则让下一个集合开始执行第 1 步。
// 注： NP完全问题，不存在多项式算法
int my_sum(vector<int>& nums){
    int sum = 0;
    for(int i:nums){
        sum = sum + i;
    }
    return sum;
}
bool backtrack(int k, int start, int target, int sum, vector<int> used, vector<int>& nums){
    if(k==0){
        return true;
    }
    if(sum==target){
        return backtrack(k-1, 0, target, 0, used, nums);
    }

    for(int i = start;i<nums.size();i++){
        if(used[i]==1){
            continue;
        }
        if(sum + nums[i]>target){
            continue;
        }
        used[i] = 1;
        sum = sum + nums[i];
        if(backtrack(k, i+1, target, sum, used, nums)){
            return true;
        }
        sum = sum - nums[i];
        used[i] = 0;
    }
    return false;
}
bool canPartitionKSubsets(vector<int>& nums, int k) {
    if((my_sum(nums)%k)!=0){
        return false;
    }
    int target = my_sum(nums)/k;
    vector<int> used(nums.size());
    return backtrack(k, 0, target, 0, used, nums);
} // 时间复杂度 O(n*target*k)


// 4. 形成3的最大倍数
// 题目描述：给你一个整数数组 digits，你可以通过按任意顺序连接其中某些数字来形成 3 的倍数，请你返回所能得到的最大的 3 的倍数。
// 思路：暴力解法，直接回溯找出所有的3的倍数的排列，复杂度太高
//   找规律解法：能被3整除表示所有位数加在一起是3的倍数，定义合为sum，那么考虑如下情况：
//      sum%3 == 0: 那么sum就是结果
//      sum%3 == 1: 那么就是移除一个(%3==1)的数字，或者2个(%3==2)的数字
//      sum%3 == 2: 那么移除一个(%3==2)的数字 或者2个(%3==1)的数字
int sum = 0; //数字字面和
int cnt[10]; //每个数字出现的频率
// CanDelete函数判断是否能移除某个（%3结果为1或者2）的数字（）
bool CanDelete(int x){
    for (int i = x; i <= 9; i += 3){ // 1，4，7或2，5，8
        if (cnt[i] > 0){ // 存在这莫个数字
            --cnt[i]; // 更新这个数字的计数器
            return true;
        }
    }
    return false;
}
string largestMultipleOfThree(vector<int>& digits) {
    memset(cnt, 0, sizeof(int)*10);
    for (int digit: digits){
        // cout << digit << endl;
        ++cnt[digit];
        sum += digit;
    }
    // 判断模的结果
    int m = sum % 3;
    if (m == 1){
        if (!CanDelete(1)){
            // 不能移除1，那么就移除两个2
            CanDelete(2);
            CanDelete(2);
        }
    }
    else if (m == 2){
        if (!CanDelete(2)){
            CanDelete(1);
            CanDelete(1);
        }
    }

    string res = "";
    // 所有数字出现的次数保存在cnt里，现在只需要倒序去构建尽可能大的数字
    for (int i = 9; i >= 0; --i){
        while (cnt[i] > 0){
            --cnt[i];
            res += '0' + i;
        }
    }
    // 特殊情况 首个字符为0，即都为0
    if (!res.empty() && res[0] == '0'){
        return "0";
    }
    return res;
}


//  5. 可能组成目标和的表达式数目
int result = 0;

/* 主函数 */
int findTargetSumWays(int[] nums, int target) {
    if (nums.length == 0) return 0;
    backtrack(nums, 0, target);
    return result;
}

/* 回溯算法模板 */ // 更美观直白的写法
void backtrack(int[] nums, int i, int remain) {
    // base case
    if (i == nums.length) {
        if (remain == 0) {
            // 说明恰好凑出 target
            result++;
        }
        return;
    }
    // 给 nums[i] 选择 - 号
    remain += nums[i];
    // 穷举 nums[i + 1]
    backtrack(nums, i + 1, remain);
    // 撤销选择
    remain -= nums[i];

    // 给 nums[i] 选择 + 号
    remain -= nums[i];
    // 穷举 nums[i + 1]
    backtrack(nums, i + 1, remain);
    // 撤销选择
    remain += nums[i];
}

// 更顺畅的写法
int count = 0;
void backtrack_sumways(vector<int>& nums, int target, int index, int sum) {
    if (index == nums.size()) {
        if (sum == target) {
            count++;
        }
    } else {
        backtrack_sumways(nums, target, index + 1, sum + nums[index]);
        backtrack_sumways(nums, target, index + 1, sum - nums[index]);
    }
}
int findTargetSumWays(vector<int>& nums, int target) {
    backtrack_sumways(nums, target, 0, 0);
    return count;
}

// 6. 找所有递增子序列
// 题目描述：返回所有该数组中不同的递增子序列，递增子序列中 至少有两个元素。
//   数组中可能含有重复元素，如出现两个整数相等，也可以视作递增序列的一种特殊情况。
class SolutionFindSubsequences {
public:
    vector<vector<int>> ans;

    void backtrack(vector<int>& nums, int curIdx, int lastnum, vector<int>& track) {
        if (curIdx == nums.size()) {
            if (track.size() >= 2) {
                ans.push_back(track);
            }
            return;
        }
        // 选择当前元素
        if (nums[curIdx] >= lastnum) {
            track.push_back(nums[curIdx]);
            backtrack(nums, curIdx + 1, nums[curIdx], track);
            track.pop_back();
        }
        // 不选择当前元素 （这里是难点，要保证去重）
        //  这里给「不选择」做一个限定条件，只有当当前的元素不等于上一个选择的元素的时候，才考虑不选择当前元素，直接递归后面的元素
        // 相当于 若相等，则不能不选当前数字, 若不等，方可考虑不选当前数字。 比如你可以模拟以下5666
        if (nums[curIdx] != lastnum) {
            backtrack(nums, curIdx + 1, lastnum, track);
        }
    }

    vector<vector<int>> findSubsequences(vector<int>& nums) {
        vector<int> path;
        backtrack(nums, 0, INT_MIN, path);
        return ans;
    }
}; // 时间复杂度：O(n*2^n)
// 空间：O(n)
#endif //DATASTRUCT_ALGORITHM_BACKTRACK_H

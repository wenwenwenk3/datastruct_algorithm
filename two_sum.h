//
// Created by kai.chen on 2021/12/4.
// 1.两数之和
// 2.三数之和
// 3.100数之和
//      变体1.1 设计一个类,实现类似数据流中求两数之和是否存在
//      变体2.1 最接近k的的三数之和
//      变体3.1 和为k的子数组   见array_string_op 3.
//      变体3.2 子数组最小值之和
// 4.整数拆分使乘积最大化
// 5.两数相加(链表求和)
// 6.字符串相加、36进制加法
// 7. rand7 生成 rand10
//
// 8. 公平的糖果交换
#ifndef DATASTRUCT_ALGORITHM_TWO_SUM_H
#define DATASTRUCT_ALGORITHM_TWO_SUM_H
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <vector>
#include <list>
using namespace std;

// 1.两数之和
//  基本做法：考察哈希表处理问题（将数组下标当成value，元素当成key）建立哈希表时间复杂度：O(N), 空间：O(N)
//  暴力解法直接用两个下标变量两个遍历，时间复杂度：O(N2)，空间：O(1)
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> hmap;
    for(int i = 0; i < nums.size(); i++){
        hmap[nums[i]] = i;
    }

    for(int i = 0; i < nums.size(); i++){
        if(hmap.count(target - nums[i]) && hmap[target - nums[i]] != i){
            return vector<int>{i, hmap[target - nums[i]]};
        }
    }
    return vector<int>{-1,-1};
}

// 返回所有和为target的元素对，不能出现重复
//  如[1,3,1,2,2,3], target=4的正确结果为[[1,3],[2,2]]， [1,3]和[3,1]就算重复
vector<int> two_sum_in_sorted_array(vector<int>& nums, int target) {
    int left = 0, right = nums.size()-1;
    while(left < right){
        if(left + right == target){
            return vector<int>{nums[left], nums[right]};
        }
        else if(left + right > target){
            right--;
        }
        else if(left + right > target){
            left++;
        }
    }
    return vector<int>{-1,-1};
}

vector<vector<int>> two_sum_in_duplicated(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());

    vector<vector<int>> res;
    int left = 0, right = nums.size()-1;
    while(left < right){
        int sum = nums[left] + nums[right];
        int left_value = nums[left], right_value = nums[right];
        if(sum == target){
            res.push_back(vector<int>{left_value, right_value});
            // 保证每个结果只添加一次
            while(left < right && nums[left] == left_value) left++;
            while(left < right && nums[right] == right_value) right--;
        }
        else if(sum < target){
            while(left < right && nums[left] == left_value) left++;
        }
        else if(sum > target){
            while(left < right && nums[right] == right_value) right--;
        }
    }

    return res;
}// 时间复杂度是: 双指针操作O(N),排序O(N*logN) ->总的时间复杂度为O(N*logN)


// 2.三数之和
// 题目描述：找出满足条件"三个数a+b+c=0", 且不重复的三元组
vector<vector<int>> threeSumTarget(vector<int>& nums, int target);
vector<vector<int>> threeSum(vector<int>& nums) {
    return threeSumTarget(nums, 0);
}

vector<vector<int>> twoSumTarget(vector<int>& nums, int start, int target) {
    // 稍微改造一下上面的twoSum，让左指针从start开始，其他不变
    vector<vector<int>> res;
    int left = start, right = nums.size()-1;
    while(left < right){
        int sum = nums[left] + nums[right];
        int left_value = nums[left], right_value = nums[right];
        if(sum == target){
            res.push_back(vector<int>{left_value, right_value});
            // 保证每个结果只添加一次
            while(left < right && nums[left] == left_value) left++;
            while(left < right && nums[right] == right_value) right--;
        }
        else if(sum < target){
            while(left < right && nums[left] == left_value) left++;
        }
        else if(sum > target){
            while(left < right && nums[right] == right_value) right--;
        }
    }
    return res;
}

vector<vector<int>> threeSumTarget(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());
    int n = nums.size();
    vector<vector<int>> res;

    // 穷举第一个数
    for(int i = 0; i < n; i++){
        // 对target-nums[i]算twoSum
        vector<vector<int>> tups = twoSumTarget(nums, i+1, target-nums[i]);
        // 若找到满足条件的二元祖，再加上nums[i]就是答案
        for(auto& tup: tups){
            tup.push_back(nums[i]);
            res.push_back(tup);
        }
        // 和twoSum的做法一样，跳过第一个数字重复的情况
        while(i<n-1 && nums[i] == nums[i+1]) i++;
    }
    return res;
} // O(N^2)


void test_three_sum(){
    int a[] = {-1,0,1,2,-1,-4};
    vector<int> vec(a, a + sizeof(a)/sizeof(a[0]));

    vector<vector<int>> res = threeSum(vec);


    int sz = res.size();
    for(int i = 0; i < sz; i++){
        printf("%d, %d, %d\n",res[i][0], res[i][1], res[i][2]);
    }
}


// 3.100Sum问题
vector<vector<int>> nSumTarget(vector<int>& nums, int n, int start, int target){
    // 在进入之前先sort
    int sz = nums.size();
    vector<vector<int>> res;
    if(n < 2 || sz < n){ //最少应该是2sum，且数组大小不能小于n
        return res;
    }

    if(n == 2){ // 2Sum是base case
        int left = start, right = nums.size()-1;
        while(left < right){
            int sum = nums[left] + nums[right];
            int left_value = nums[left], right_value = nums[right];
            if(sum == target){
                res.push_back(vector<int>{left_value, right_value});
                // 保证每个结果只添加一次
                while(left < right && nums[left] == left_value) left++;
                while(left < right && nums[right] == right_value) right--;
            }
            else if(sum < target){
                while(left < right && nums[left] == left_value) left++;
            }
            else if(sum > target){
                while(left < right && nums[right] == right_value) right--;
            }
        }
    }
    else{ // n > 2时，递归计算(n-1)Sum的结果
        for(int i = start; i < sz; i++){
            vector<vector<int>> tups = nSumTarget(nums,n-1, i+1, target-nums[i]);
            for(vector<int>& tup :tups){
                tup.push_back(nums[i]);
                res.push_back(tup);
            }
            // 和2sum一样，跳过重复数字（若有需要）
            while(i < sz-1 && nums[i] == nums[i+1]) i++;
        }
    }
    return res;
}

// 变体1.1
// 设计一个类,实现类似数据流中求两数之和是否存在
//   方法一：用map保存出现的频率
//     复杂度add方法是O(1),  find方法是O(N)，空间复杂度是O(N)
//     （适合add场景多，find场景少）
class TwoSum{
private:
    unordered_map<int, int> freq;
public:
    void add(int num){
        freq[num] += 1;
    }
    bool find(int target){
        for(auto key: freq){
            int other = target - key.first;
            if(other == key.first && freq[other] > 1){
                return true;
            }
            if(other != key.first && freq.count(other)){
                return true;
            }
        }
        return false;
    }
};

//   方法二：用set保存出现的数的可能和
//     复杂度add方法是O(N),  find方法是O(1)，空间复杂度是O(2^N)
//     （适合find场景多，add场景少,
//          缺点是最坏情况每次add后sum的大小都会翻一倍，空间复杂度是O(2^N)。只有在数据非常小才会用）
class TwoSum_V2{
private:
    unordered_set<int> sumSet; //
    list<int> datalist;
public:
    void add(int num){
        for(auto i : datalist){
            sumSet.insert(i+num);
        }
        datalist.push_back(num);
    }
    bool find(int val){
        return sumSet.count(val);
    }
};

// 2.1. 最接近的三数之和
int threeSumClosest(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());

    int sz = nums.size();
    int closest = INT_MAX; // 如果可能在abs时越界，尝试用1e9
    // 穷举第一个元素
    for(int i = 0; i < sz; i++){
        // 和上一个元素相等，直接跳过
        if(i > 0 && nums[i] == nums[i-1]) continue;

        int left = i+1, right = sz-1;
        while(left < right){
            int sum = nums[i] + nums[left] + nums[right];
            int left_value = nums[left], right_value = nums[right];

            if(abs(sum-target) < abs(closest-target)){
                closest = sum; // 更新closest
            }
            if(sum == target) return target; // 这里可以找到直接退出(小优化)
            else if(sum < target){
                while(left < right && nums[left] == left_value) left++;
            }
            else if(sum > target){ // 向左移动right指针，移动到下一个不相等的位置
                while(left < right && nums[right] == right_value) right--;
            }
        }
    }
    return closest;
} // 时间复杂度O(N^2), 排序是O(n*logN), 一重循环O(N)，双指针运动O(N).
  // 空间复杂度O(N), 排序需要logN，如果要求不能修改nums数组，也需要O(N)空间拷贝





// 4. 整数拆分
// 题目描述：给定一个正整数 n(n>=2, n<=58)，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。
// 思路：
//   dp[i]表示正整数拆分之后的最大乘积
//      将 i 拆分成 j 和 i−j 的和，且 i-j 不再拆分成多个正整数，此时的乘积是 j×(i−j)；
//      将 i 拆分成 j 和 i−j 的和，且 i-j 继续拆分成多个正整数，此时的乘积是 j×dp[i−j]。
//   所以, 当固定j时：dp[i] = max(j×(i−j), j×dp[i−j]), 而j的范围是[1,i-1]
//   即dp转移方程为：dp[i] = for(j = 1..i-1) max(j×(i−j), j×dp[i−j])
int integerBreak(int n) {
    vector <int> dp(n + 1);
    for (int i = 2; i <= n; i++) {
        int curMax = 0;
        for (int j = 1; j < i; j++) {
            curMax = max(curMax, max(j * (i - j), j * dp[i - j]));
        }
        dp[i] = curMax;
    }
    return dp[n];
} // 时间复杂度O(N^2)


// 5.两数相加
// 有2个逆序的链表，从低位开始相加，相加结果按位依次赋给输出链表，最终返回输出链表头指针，需要注意的是各种进位问题：普通进位和最终位进位。
ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2){
    ListNode *p1 = l1, *p2 = l2, *p = new ListNode;
    p->next = nullptr;
    ListNode *dummy = p;
    int a, b, sum, flag = 0; // flag进位标志
    while(p1 != nullptr || p2 != nullptr){
        a = p1 ? p1->val: 0;
        b = p2 ? p2->val: 0;

        sum = a + b + flag;
        flag = sum/10;

        p->next = new ListNode;
        p->next->val = sum%10;
        p->next->next = nullptr;

        p = p->next;

        if(p1 != nullptr) {
            p1 = p1->next;
        }
        if(p2 != nullptr){
            p2 = p2->next;
        }
    }
    if(flag != 0){
        p->next = new ListNode;
        p->next->val = 1;
        p->next->next = nullptr;
    }
    return dummy->next;
}
// 两数相加II， 翻转转化为上面解法

// 6. 字符串相加
// 从后往前加，每次记录进位。注意最终进位，最后加成的字符串翻转就是结果
string addString(string num1, string num2){
    int carry = 0;
    int i = num1.size()-1, j = num2.size()-1;
    string res;
    while(i >= 0 || j >=0 || carry){ // 有最终进位还得加上
        int x = i >= 0 ? num1[i]-'0' : 0;
        int y = j >= 0 ? num2[j]-'0' : 0;
        int temp = x + y + carry;

        res += ('0' + temp%10);
        carry = temp/10;
        i-- , j--;
    }
    reverse(res.begin(), res.end());
    return res;
}
// 6. 36进制加法
int getInt(char c){
    // 输入保证c为合法36进制数
    if('0' <= c & c <= '9'){  // 0-9
        return c - '0';
    }else{ // a-z
        return c - 'a' + 10;
    }
}
char getChar(int x){
    if(x <= 9){
        return x + '0';
    }else{
        return x -10 + 'a';
    }
}

string add36String(string num1, string num2){
    int carry = 0;
    int i = num1.size()-1, j = num2.size()-1;
    string res;
    while(i >= 0 || j >=0 || carry){
        int x = i >= 0 ? getInt(num1[i]) : 0;
        int y = j >= 0 ? getInt(num2[j]) : 0;
        int temp = x + y + carry;
        res += getChar(temp%36);
        carry = temp/36;
        i-- , j--;
    }
    reverse(res.begin(), res.end());
    return res;
}


// 7. rand7 生成 rand10
// 思路：证明： rand()7能等概率生成1~7；
// rand7() - 1能等概率生成0~6；
// (rand7() - 1) * 7能等概率生成{0, 7, 14, 21, 28, 35, 42}；
// (rand7() - 1) * 7 + rand7()能等概率生成1~49。
int rand7(){ return 5;}
int rand10() {
    while (true) {
        int ans = (rand7() - 1) * 7 + (rand7() - 1); // 进制转换
        if (1 <= ans && ans <= 10) return ans;
    }
}


// 8. 公平的糖果交换
// 输入：aliceSizes = [1,2], bobSizes = [2,3]
//输出：[1,2]
//
// 思路： 记爱丽丝的糖果棒的总大小为 sumA，鲍勃的糖果棒的总大小为 sumB。设答案为 {x,y}，即爱丽丝的大小为 x 的糖果棒与鲍勃的大小为 y 的糖果棒交换，
// 则有如下等式：sumA−x+y = sumB+x−y
//      化简得：x = y + (sumA+sumB)/2
// 所以，只要对于 bobsize任意一个y，存在一个x满足上述条件即为一个可行解
vector<int> fairCandySwap(vector<int>& aliceSizes, vector<int>& bobSizes) {
    int sumA = 0;
    for(auto& i: aliceSizes){
        sumA += i;
    }
    int sumB = 0;
    for(auto& i: bobSizes){
        sumB += i;
    }

    unordered_set<int> set(aliceSizes.begin(), aliceSizes.end());

    vector<int> ans;
    for (auto& y : bobSizes) {
        int x = y + (sumA - sumB) / 2;

        if (set.count(x)) {
            ans = vector<int>{x, y};
            break;
        }
    }
    return ans;
}


// 9. k-diff数对
// 题目描述：k-diff数对满足条件，求不同的k-diff数对的数目
//  0 <= i, j < nums.length
//  i != j
//  |nums[i] - nums[j]| == k
// 思路：遍历数组，每次判断 j 左侧是否有满足条件的 i
int findkPairs(vector<int>& nums, int k) {
    unordered_set<int> visited;
    unordered_set<int> res;
    for (const int& num : nums) {
        if (visited.count(num - k)) {
            res.insert(num - k);
        }
        if (visited.count(num + k)) {
            res.insert(num);
        }
        visited.insert(num);
    }
    return res.size();
}


#endif //DATASTRUCT_ALGORITHM_TWO_SUM_H

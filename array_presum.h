//
// Created by kai.chen on 2022/6/7.
//
//     1. 区域和检索
//        1.1 二维区域和检索
//     2. 需要补充粉笔的学生编号
//     3. 蜡烛之间的盘子
//
//     4. 从链表中删去总和值为零的连续节点
//     5. 0和1的个数相同的子数组
//
//     6. 使数组和能被 P 整除
//
#define LL long long
#ifndef DATASTRUCT_ALGORITHM_ARRAY_PRESUM_H
#define DATASTRUCT_ALGORITHM_ARRAY_PRESUM_H
#include <vector>
using namespace std;


// 1. 区域和检索
// 题目描述：求出数组从索引 i 到 j（i ≤ j）范围内元素的总和，包含 i、j 两点。
// 思路：前缀和
// 时间复杂度：初始化 O(n)，每次检索 O(1)
class NumArray {
private:
    vector<int> sums;
public:
    explicit NumArray(vector<int>& nums) {
        int n = nums.size();
        sums.resize(n + 1);
        for (int i = 0; i < n; i++) {
            sums[i + 1] = sums[i] + nums[i];
        }
    }

    int sumRange(int i, int j) { // [i, j+1)
        return sums[j + 1] - sums[i];
    }
};


// 1.1 二维区域和检索
// 思路：二维前缀和
// 时间复杂度：初始化 O(mn)，每次检索 O(1)
class NumMatrix2 {
    vector<vector<int>> presums;
public:
    NumMatrix2(vector<vector<int>>& matrix) {
        int n = matrix.size(), m = n == 0 ? 0 : matrix[0].size();
        // 与「一维前缀和」一样，前缀和数组下标从 1 开始，因此设定矩阵形状为 [n + 1][m + 1]（模板部分）
        presums.resize(n+1, vector<int>(m+1,0));
        // 预处理出前缀和数组（模板部分)
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                presums[i+1][j+1] = presums[i][j+1] + presums[i+1][j] - presums[i][j] + matrix[i][j];
            }
        }
    }

    int sumRegion(int x1, int y1, int x2, int y2) {
        // 求某一段区域和 [i, j] 的模板是 presum[x2][y2] - sum[x1 - 1][y2] - sum[x2][y1 - 1] + sum[x1 - 1][y1 - 1];（模板部分）
        // 但由于我们源数组下标从 0 开始，因此要在模板的基础上进行 + 1
        x1++; y1++; x2++; y2++;
        return presums[x2][y2] - presums[x1-1][y2] - presums[x2][y1-1] + presums[x1-1][y1-1];
    }
};



// 2. 需要补充粉笔的学生编号
// 暴力法
int chalkReplacer1(vector<int>& chalk, int k) {
    int i = -1;
    while(k>=0){
        i = (i+1) % chalk.size();
        k -= chalk[i];
    }
    return i;
}
// 前缀和优化
int chalkReplacer(vector<int>& chalk, int k) {
    int n = chalk.size();
    vector<LL> presum(n+1, 0);
    for(int i = 1; i<n+1; i++){
        presum[i] = presum[i-1]+chalk[i-1];
    }
    k %= presum[n];
    int i;
    for(i = 1; i < n+1; i++){
        if(presum[i] > k){
            break;
        }
    }
    return i-1;
}

// 3. 蜡烛之间的盘子
// 题目描述：求子字符串中 在 两支蜡烛之间 的盘子的 数目 。如果一个盘子在 子字符串中 左边和右边 都 至少有一支蜡烛，那么这个盘子满足在 两支蜡烛之间 。
//      比方说，s = "||**||**|*"，查询[3, 8]，表示的是子字符串"*||**|"。子字符串中在两支蜡烛之间的盘子数目为2
//          子字符串中右边两个盘子在它们左边和右边 都 至少有一支蜡烛
// 思路：假设 需要查的区间为[a, b]，只需要找到 [c, d]（c是a右边第一个蜡烛位置，d是b左边第一个蜡烛位置）之前的盘子数目，盘子数可以用前缀和存储
vector<int> platesBetweenCandles(string s, vector<vector<int>>& queries){
    int n = s.length();
    vector<int> presums(n+1, 0);
    for(int i = 0; i < n; i++){
        presums[i+1] = presums[i] + (s[i] == '*' ? 1 : 0);
    }
    vector<int> l(n); // 记录每个位置左边第一个蜡烛位置
    vector<int> r(n); // 记录每个位置右边第一个蜡烛位置
    for (int i = 0, j = n - 1, p = -1, q = -1; i < n; i++, j--) {
        if (s[i] == '|') p = i;
        if (s[j] == '|') q = j;
        l[i] = p; r[j] = q;
    }

    vector<int> res(queries.size(), 0);
    int i = 0;
    for(const auto& query: queries){
        int a = query[0], b = query[1];
        int c = r[a], d = l[b];

        if(c != -1 && d != -1 && c <= d){
            res[i] = presums[d+1] - presums[c];
        }
        i++;
    }
    return res;
} // 时间复杂度：O(n+q), n是字符串s的长度，q是查询的次数
// 空间复杂度：O(n)

void test_platesBetweenCandles(){
    vector<vector<int>> q = {{2,5},{5,9}};
    auto res = platesBetweenCandles("**|**|***|", q);
    for_each(res.begin(), res.end(), [](const int it){
        cout<<it<<",";
    });
}


// 4. 从链表中删去总和值为零的连续节点
// 思路：一边遍历链表，一遍记录当前的和，如果之前存在这个和，那么就把他们中间的节点都断掉
//     比如以 [1, 2, 3, -3, 4] 为例，其前缀和数组为 [1, 3, 6, 3, 7] ，我们发现有两项均为 3，则 6 和 第二个 3 所对应的原数组中的数字是可以消掉的。换成链表其实也是一样的思路，
//       把第一个 3 的 next 指向第二个 3 的 next 即可
ListNode* removeZeroSumSublists(ListNode* head){
    ListNode* dummy = new ListNode(0); // dummy 避免头节点被删需要单独考虑
    dummy->next = head;
    unordered_map<int, ListNode*> preSum;
    int curSum = 0;
    for(ListNode* cur = dummy; cur != nullptr; cur = cur->next){
        curSum += cur->val;
        if(preSum.count(curSum)){
            ListNode* tmp = preSum[curSum]->next;
            preSum[curSum]->next = cur->next; // 截断连接
            int tmpsum = curSum;
            for(;tmp != cur; tmp = tmp->next){ // 从map中删除对应区间和
                tmpsum += tmp->val;
                preSum.erase(tmpsum);
            }

        }else{
            preSum[curSum] = cur;
        }
    }
    return dummy->next;
}
/* go 版本
 * func removeZeroSumSublists(head *ListNode) *ListNode {
    //添加一个哑结点防止头结点被删除等复杂情况
    dummy := &ListNode{Val :0, Next : head}
    sum := 0
    //创建一个存储节点和sum的哈希表
    HasMap := make(map[int]*ListNode)
    for p := dummy; p != nil; p = p.Next {
        //将每个节点的值进行累加
        sum += p.Val
        //本节点及其前面节点的val累加值放入key
         //用一个hash表map来存储每个结点的前缀和，如果有前缀和相等的节点自动覆盖为后面的节点
        HasMap[sum] = p
    }
    sum2 := 0
    for p := dummy; p != nil; p = p.Next {
        sum2 += p.Val
        //哈希表里存的这个sum 肯定是第二次出现的
        p.Next = HasMap[sum2].Next
    }
    return dummy.Next

}
 * */

// 5. 0和1的个数相同的子数组
//性质：子数组长度必然为偶数，且长度至少为 2。
//具体的，我们在预处理前缀和时，将 nums[i] 为 0 的值当做 −1 处理。
// 问题转化为：如何求得最长一段区间和为 0 的子数组。 同时使用「哈希表」来记录「某个前缀和出现的最小下标」是多少。
int findMaxLength(vector<int>& nums) {
    int n = nums.size(), ans = 0;
    vector<int> sum(n + 1);
    for (int i = 1; i <= n; i++) sum[i] = sum[i - 1] + (nums[i - 1] == 0 ? -1 : 1);
    unordered_map<int, int> mp;
    mp[0] = 0;
    for (int i = 1; i <= n; i++) {
        int t = sum[i];
        if (mp.count(t)) ans = max(ans, i - mp[t]);
        if (!mp.count(t)) mp[t] = i;
    }
    return ans;
}


#endif //DATASTRUCT_ALGORITHM_ARRAY_PRESUM_H

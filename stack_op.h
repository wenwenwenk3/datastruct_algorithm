//
// Created by kai.chen on 2021/12/29.
//
//      1. 最小栈
//      2. 两个栈组成一个队列
//      3. 单调栈
//      4. 区间最小值乘区间和 的最大值 monstack.h 1.8.
//
//      5. 栈的压入、弹出序列
//      6. 最大频率栈
//      7. 双栈排序
//
//      8. 根据身高重建队列
//
#ifndef DATASTRUCT_ALGORITHM_STACK_OP_H
#define DATASTRUCT_ALGORITHM_STACK_OP_H
#include <stack>
#include <queue>
using namespace std;

// 1. 最小栈
// 题目描述：实现一个特殊的栈，能在O(1)时间获得最小值，O(1)pop、push
// 思路：一个正常的栈stack + 一个保存每一步最小值的栈stackMin
//      具体有两种做法：
//  做法一： push规则：先压入数据栈，然后判断最小栈是否为空
//            如果为空，直接将当前数据压入最小栈
//            如果不为空，判断newNum是否比栈顶小或者相等。如果小或相等就压入最小栈
//          eg:数据为[3,4,5,1,2,1], 栈[1,2,1,5,4,3], 最小栈[1,null,1,null,null,3] => [1,1,3]
//      pop规则：先弹出数据栈栈顶元素value，显然value一定大于等于stackMin栈顶。
//            如果等于stackMin栈顶，就弹出一次栈顶。否则不用动stackMin
//     getMin规则：显然stackMin栈顶始终记录着最小值
//  做法二：push规则：重复压入，如果不为空，判断newNum是否小于等于stackMin栈顶
//          是的话，压入最小栈。不是的话把栈顶(最小栈的最小值)再压一个在栈顶
//          eg:数据为[3,4,5,1,2,1], 栈[1,2,1,5,4,3], 最小栈[1,1,1,3,3,3] => [1,1,1,3,3,3]
// 做法一和做法二都是时间复杂度O(1),空间复杂度O(n)
//  但区别是：做法一stackMin压入时更节省空间，但是弹出更费时间
//      而做法二stackMin压入时更费空间，但是弹出时更省时间
class StackMin {
private:
    stack<int> stk;
    stack<int> stkMin;
public:
    // StackMin(){stk = new stack<int>()}
    void push(int newNum){
        stk.push(newNum);
        if(stkMin.empty()){
            stkMin.push(newNum);
        }
        else if(newNum <= stkMin.top()){
            stkMin.push(newNum);
        }
    }
    int pop(){
        if(stk.empty()) exit(-1);
        int value = stk.top();
        stk.pop();
        if(value == stkMin.top()){
            stkMin.pop();
        }
        return value;
    }
    int getMin(){
        if(stkMin.empty()) exit(-1);
        return stkMin.top();
    }
    void push_v2(int newNum){
        stk.push(newNum);
        if(stkMin.empty()){
            stkMin.push(newNum);
        }
        else if(newNum <= stkMin.top()){
            stkMin.push(newNum);
        }
        else{
            int curMin = getMin();
            stkMin.push(curMin);
        }
    }
    int pop_v2(){
        if(stk.empty()) exit(-1);
        int value = stk.top();
        stk.pop();
        stkMin.pop();
        return value;
    }
};


// 2. 两个栈组成队列
// 题目描述：用两个栈支持队列的push,pop
// 思路：将两个栈分成：一个压入栈，一个弹出栈
//
//   { 若需要左神的更高效做法：需要注意的就是：
//      (1)压入栈满了，往弹出栈倒数据必须一次性把压入栈的数据全部压入弹出栈
//      (2)弹出栈不为空，压入栈不能往弹出栈压入数据
//   }
class TwoStackQueue{
    // 插入元素: stack1 直接插入元素
    void pushTail(int newNum){
        stk1.push(newNum);
    }
    // 弹出元素:
    int popFront(){
        // 检查弹出栈stk2是否为空
        if(stk2.empty()){
            // 把压入栈stk1倒到弹出栈stk2
            while (!stk1.empty()) {
                stk2.push(stk1.top());
                stk1.pop();
            }
        }
        if(stk2.empty()){ // 倒完还是空，说明没有元素
            return -1;
        }else{
            int topValue = stk2.top();
            stk2.pop();
            return topValue;
        }
    }
    // 清零
    void CQueue() {
        while (!stk1.empty()) {
            stk1.pop();
        }
        while (!stk2.empty()) {
            stk2.pop();
        }
    }
    // 判断是否为空
    bool empty() {
        // 检查弹出栈stk2是否为空
        if(stk2.empty()){
            // 把压入栈stk1倒到弹出栈stk2
            while (!stk1.empty()) {
                stk2.push(stk1.top());
                stk1.pop();
            }
        }
        return stk2.empty(); // 倒完还是空，说明没有
    }
private:
    stack<int> stk1; // 压入栈stk1
    stack<int> stk2; // 弹出栈stk2
};

// 思考两个队列实现一个栈 :
// 入栈操作时，首先将元素入队到 queue2，然后将 queue1的全部元素依次出队并入队到 queue2，
// 此时 queue2 的前端的元素即为新入栈的元素，再将 queue1和 queue2互换，
// 则 queue1 的元素即为栈内的元素，queue1的前端和后端分别对应栈顶和栈底。
class MyStack {
public:
    queue<int> q1; // 存储栈内的元素
    queue<int> q2; // 入栈操作的辅助队列

    /** Initialize your data structure here. */
    MyStack() {}

    /** Push element x onto stack. */
    void push(int x) {
        q2.push(x);
        while (!q1.empty()) {
            q2.push(q1.front());
            q1.pop();
        }
        swap(q1, q2);
    }

    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int r = q1.front();
        q1.pop();
        return r;
    }

    /** Get the top element. */
    int top() {
        int r = q1.front();
        return r;
    }

    /** Returns whether the stack is empty. */
    bool empty() {
        return q1.empty();
    }
}; // 时间复杂度：入栈操作 O(n)，其余操作都是 O(1)，其中 n 是栈内的元素个数。

// 一个队列实现栈： 原地将元素反过来push一遍
class MyStack2 {
public:
    queue<int> q;

    /** Initialize your data structure here. */
    MyStack2() {}

    /** Push element x onto stack. */
    void push(int x) {
        int n = q.size();
        q.push(x);
        for (int i = 0; i < n; i++) {
            q.push(q.front());
            q.pop();
        }
    }

    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int r = q.front();
        q.pop();
        return r;
    }
    /** Get the top element. */
    int top() {
        return q.front();
    }
    /** Returns whether the stack is empty. */
    bool empty() {
        return q.empty();
    }
}; //时间复杂度：入栈操作 O(n)O(n)，其余操作都是 O(1)O(1)


// 3. 单调栈 见 数据结构进阶 advanced


// 5. 栈的压入、弹出序列
// 题目描述：输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
//      输出：true
// 注：题目指明 pushed 是 popped 的排列 。因此，无需考虑 pushed 和 popped 长度不同 或 包含元素不同 的情况。
bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
    stack<int> stk; // 用一个栈模拟压入顺序
    int t = 0;
    int n = popped.size();
    for(auto x : pushed){
        stk.push(x); // 依次压入 pushed 序列
        // 如果压入的元素等于弹出序列的首个元素，栈 pop出这个压入元素
        // 如果 不等于就只能继续压 压到出现等于
        while(t < n && !stk.empty() && popped[t] == stk.top()){ // 当然pop之前注意 stk为空
            stk.pop();
            t++;
        }
    }
    return stk.empty();
}

void test_validateStackSequences(){
    vector<int> pushed = {1,0};
    vector<int> poped = {1,0};
    if (validateStackSequences(pushed, poped)){
        cout << "True" << endl;
    }else {
        cout << "False" <<endl;
    }

}

// 6. 最大频率栈
// 题目描述：设计一个类似堆栈的数据结构，将元素推入堆栈，并从堆栈中弹出出现频率最高的元素。
//  FreqStack() 构造一个空的堆栈。
//  void push(int val) 将一个整数val压入栈顶。
//  int pop() 删除并返回堆栈中出现频率最高的元素。
//      如果出现频率最高的元素不只一个，则移除并返回最接近栈顶的元素。
// 思路：显然，我们更关心元素的频率。令 freq 作为 x 到 x 的出现次数的映射 Map。
//      此外，我们也（可能）关心 maxfreq，即栈中任意元素的当前最大频率。这是理所应当的事情，因为我们必须弹出频率最高的元素。
//    那么当前主要的问题就变成了：在具有相同的（最大）频率的元素中，怎么判断那个元素是最新的？
//        我们可以使用栈来查询这一信息：靠近栈顶的元素总是相对更新一些。
class FreqStack {
private:
    unordered_map<int, int> freq; // 记录每个元素出现的频率
    unordered_map<int, stack<int>> record; // 记录每个频率下的元素，先进后出
    int maxfreq; // 最大频率
public:
    FreqStack():maxfreq(0){}

    void push(int val) {
        freq[val]++;
        if(freq[val] > maxfreq){
            maxfreq = freq[val];
        }
        record[freq[val]].push(val);
    }

    int pop() {
        int x = record[maxfreq].top();
        freq[x]--;
        record[maxfreq].pop();
        if(record[maxfreq].empty()){
            record.erase(maxfreq);
            maxfreq--;
        }
        return x;
    }
};


// 7. 双栈排序
// 题目描述：设计算法给一个乱序栈排序，要求只能使用一个辅助栈
// 思路：维护辅助栈为单调递增栈。也就是 保证“倒腾”过程的任何时候，辅助栈的元素都是从小到大排序的！
//  用[4, 2, 1, 3]数组举个例子，模拟一下“倒腾”的过程。
//   (1) 将3出栈，存至辅助栈
//   (2) 由于1<3不能让辅助栈升序，将1存到临时变量中。
//   (3) 把3 倒腾 到原栈，直到1可以挪到辅助栈为止
stack<int> stackSort(stack<int> &stk) {
    stack<int> tmp;
    while (!stk.empty()) {
        int peak = stk.top();
        stk.pop();
        while (!tmp.empty() && tmp.top() > peak) {
            int t = tmp.top();
            tmp.pop();
            stk.push(t);
        }
        tmp.push(peak);
    }
    return tmp;
}


// 8. 根据身高重建队列
// 题目描述：people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。
//  请你重新构造并返回输入数组people 所表示的队列。返回的队列应该格式化为数组 queue ，
//  其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。
//      输入：people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
//      输出：[[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
// 思路：
//      排序：当「高度（第一维）」不同，根据高度排升序，对于高度相同的情况，则根据「编号（第二维）」排降序。
//  排序后的好处是：在从前往后处理某个people[i] 时，我们可以直接将其放置在「当前空位序列（从左往后统计的，不算已被放置的位置）」中的
//      people[i][1]+1 位（预留了前面的people[i][1] 个位置给后面的数）。
//  优化：如何快速找到「空白序列中的第 k 个位置」，这可以通过「二分 + 树状数组」来做
class Solution {
public:
    int n;
    vector<int> btr;
    int lowbit(int x){
        return x & -x;
    }

    void add(int idx, int val){
        for(int i = idx ; i <= n; i += lowbit(i)){
            btr[i] += val;
        }
    }

    int sum(int idx){
        int res = 0;
        for(int i = idx; i > 0; i -= lowbit(i)){
            res += btr[i];
        }
        return res;
    }

    // 放在空白序列的 people[i][1] + 1 位置
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        sort(people.begin(), people.end(),[&](vector<int> & a, vector<int> & b){
            if(a[0] == b[0]){
                return a[1] > b[1];
            }
            return a[0] < b[0];
        });
        n = people.size();
        btr.resize(n + 1, 0);
        vector<vector<int>> res;

        // 快速找到 res中的 空白序列的第k个位置
        //  树状数组代表位置的使用状况（1 的前缀和），0 即为该位置可以使用
        //  空白序列的第k个位置 == 在树状数组中找到第一个位置 0 的个数大于等于 k + 1 即可
        //  0 的个数 == （位置 - 此位置之前的被使用的位置数）
        for(auto & ite : people){
            res.emplace_back(ite);
        }
        for(auto & ite : people){
            int h = ite[0], k = ite[1];
            int l = 1 , r = n;
            while( l < r){
                int mid = (l + r) >> 1;
                if(mid - sum(mid) >= k + 1){ // 如果有，那么缩小， r变形小
                    r = mid;
                }else{
                    l = mid + 1;
                }
            }
            res[r-1] = ite; // r - 1 从前缀和坐标回归res位置坐标
            add(r , 1); // 更新位置剩余的信息
        }

        return res;
    }
}; // 时间复杂度: 排序需要O(N*logN)，共要处理 n 个 people[i]，每次处理需要二分，复杂度为 O(logn)，每次二分和找到答案后需要操作树状数组，复杂度为 O(logn)
//  总时间复杂度为O(N*logN*logN), 当然这题也可以直接搜数组不用二分+树状数组，时间复杂度是N^2
// 空间复杂度: O(N)

// 将每个人按照身高从大到小进行排序,
// 当我们放入第 i 个人时：当我们放入第 i 个人时，只需要将其插入队列中，使得他的前面恰好有 k_i 个人就可以
//      第i−1 个人已经在队列中被安排了位置，他们只要站在第 i 个人的前面，就会对第 i 个人产生影响，因为他们都比第 i 个人高；
//      而第 i+1,⋯,n−1 个人还没有被放入队列中，并且他们无论站在哪里，对第 i 个人都没有任何影响，因为他们都比第 i 个人矮。
vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
    sort(people.begin(), people.end(), [](const vector<int>& u, const vector<int>& v) {
        return u[0] > v[0] || (u[0] == v[0] && u[1] < v[1]);
    });
    vector<vector<int>> ans;
    for (const vector<int>& person: people) {
        ans.insert(ans.begin() + person[1], person);
    }
    return ans;
}

#endif //DATASTRUCT_ALGORITHM_STACK_OP_H

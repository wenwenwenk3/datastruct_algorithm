//
// Created by kai.chen on 2022/1/2.
//
//      1. 跳跃游戏 (判断能够到达最后一个下标)
//      2. 跳跃游戏II (求最少的跳跃次数)
//      3. 跳跃游戏III (判断是否能够跳到值为 0 的下标处)
//
//      4. 到家的最少跳跃次数
//
#ifndef DATASTRUCT_ALGORITHM_JUMP_GAME_H
#define DATASTRUCT_ALGORITHM_JUMP_GAME_H
#include <unordered_set>
#include <vector>
using namespace std;

// 1. 跳跃游戏
// 题目描述：给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
//      数组中的每个元素代表你在该位置可以跳跃的最大长度。请判断是否能够到达最后一个下标。
//      例如：输入：nums = [2,3,1,1,4]  输出：true   解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
// 思路：
//      贪心：从x能否到y，只需要判断 x + nums[x] ≥ y，那么位置y可以到达。
//          只需要依次扫描数组中的每一个位置，并实时维护一个 最远可以到达的位置maxStep，当它大于n-1即可
bool canJump(vector<int>& nums) {
    int n = nums.size();
    int maxStep = 0;
    for (int i = 0; i < n; ++i) {
        // 只有当前位置小于等于maxStep，当前位置才是可达。否则根本不可达
        if (i <= maxStep) {
            maxStep = max(maxStep, i + nums[i]);
            if (maxStep >= n - 1) {
                return true;
            }
        }
    }
    return false;
}

// 2.  跳跃游戏II
// 题目描述： 这次的目标是使用最少的跳跃次数到达数组的最后一个位置。
//          （假设你总是可以到达数组的最后一个位置。）
//    例如：  输入: nums = [2,3,1,1,4]  输出: 2  解释: 跳到最后一个位置的最小跳跃数是 2。
// 思路：
//      贪心：
//       常规思路(1): 反向查找出发位置，
//          依次扫描每一个位置，反向找能到达该位置的最前面的位置。也就是从左到右遍历数组，选择第一个满足要求的位置。
//          时间复杂度O(n^2)
//       (2): 正向查找可到达的最大位置
//          「贪心」地进行正向查找，每次找到可到达的最远位置。每次发生了新的跳跃，增加跳跃次数.
int jumpII(vector<int>& nums) {
    int pos = nums.size() - 1;
    int steps = 0;
    while (pos > 0) {
        for (int i = 0; i < pos; i++) {
            // 从左到右第一个满足到该位置的位置 i。
            if (i + nums[i] >= pos) {
                pos = i; // 更新i，接下来要想办法怎么到达位置i
                steps++;
                break;
            }
        }
    }
    return steps;
}

int jumpII_v2(vector<int>& nums) {
    int maxPos = 0;
    int currMaxPos = 0, steps = 0;
    for (int i = 0; i < nums.size()-1; ++i) {
        // 只有当前位置小于等于maxStep，当前位置才是可达。否则根本不可达
        if (i <= maxPos) {
            maxPos = max(maxPos, i + nums[i]); // 更新 maxPos：可到达的最远位置

            if (i == currMaxPos) { // 每次发生了新的跳跃，增加跳跃次数
                currMaxPos = maxPos;
                steps++;
            }
        }
    }
    return steps;
}

// 3.  跳跃游戏III (判断是否能够跳到值为 0 的下标处)
// 题目描述：非负整数数组arr，你最开始位于该数组的起始下标start处。当你位于下标i处时，你可以跳到i + arr[i] 或者 i - arr[i]。
//          请你判断自己是否能够跳到对应元素值为 0 的 任一 下标处。
//      例如： 输入：arr = [4,2,3,0,3,1,2], start = 5  输出：true
//          解释：到达值为 0 的下标 idx=3 有以下可能方案：
//              下标 5 -> 下标 4 -> 下标 1 -> 下标 3
//              下标 5 -> 下标 6 -> 下标 4 -> 下标 1 -> 下标 3
//
// 思路：BFS搜索， 每次将 start的i + arr[i]和i - arr[i]加入队列
bool canReach(vector<int>& arr, int start) {
    if (arr[start] == 0) return true;

    int n = arr.size();
    vector<bool> visited(n);

    queue<int> q;
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int i = q.front();
        q.pop();
        // 将 i+arr[i] 加入队列
        if(i + arr[i] < n && !visited[i + arr[i]]) {
            if (arr[i + arr[i]] == 0) return true;

            q.push(i + arr[i]);
            visited[i + arr[i]] = true;
        }
        // 将 i-arr[i] 加入队列
        if (i - arr[i] >= 0 && !visited[i - arr[i]]) {
            if (arr[i - arr[i]] == 0) {
                return true;
            }
            q.push(i - arr[i]);
            visited[i - arr[i]] = true;
        }
    }
    return false;
} // 时间复杂度：O(N)，其中 NN 是数组 arr 的长度。


// 4. 到家的最少跳跃次数
// 题目描述：每次可以 往前跳恰好a个位置 或 往后跳恰好b个位置。不能跳到forbidden 数组中的位置
//      求 到家的最少跳跃次数，如果没有可行方案返回-1。家在位置 x 处
//    例如：输入：forbidden = [14,4,18,1,15], a = 3, b = 15, x = 9   输出：3
//              解释：往前跳 3 次（0 -> 3 -> 6 -> 9），跳蚤就到家了。
// 思路：
//     BFS搜索，每次将位置 i+a 或 i+b 加入队列。
//      （需要注意的是 visited 需要考虑方向。正向和反向是不同的情况）
int minimumJumps(vector<int>& forbidden, int a, int b, int x) {
    if(x == 0) return 0;
    unordered_set<int> fbset(forbidden.begin(), forbidden.end());
    unordered_set<int> visited;

    // first: index, second: direction (true: 正向，false: 反向）
    queue<pair<int, bool>> q;
    q.push(make_pair(0 + a, true)); // / 从0出发，不能跳到负整数位置，所以第一步必须往前跳一步
    if(a > 6000 || fbset.find(a) != fbset.end()) return -1;
    int steps = 1;

    while (!q.empty()) {
        int sz = q.size();
        for (int i = 0; i < sz; ++i) {
            auto cur = q.front();
            q.pop();

            int curIndex = cur.first;
            bool direction = cur.second;
            if (curIndex == x) return steps;

            // 当前是正向
            if(direction == true) {
                // 前后跳都可以
                int nextIdx = curIndex + a;
                if (nextIdx <= 6000 && fbset.find(nextIdx) == fbset.end() &&
                    visited.find(curIndex) == visited.end()) {

                    q.push(make_pair(nextIdx, true));
                    visited.insert(curIndex);
                }

                int backIdx = curIndex - b;
                if (backIdx > 0 && fbset.find(backIdx) == fbset.end() &&
                    visited.find(-curIndex) == visited.end()) {

                    q.push(make_pair(backIdx, false));
                    visited.insert(-curIndex);
                }
            }
            // 当前是反向
            else {
                // 只能向前跳，不然永远跳不到终点
                int nextIdx = curIndex + a;
                int nextDir = 1;
                if (nextIdx <= 6000 && fbset.find(nextIdx) == fbset.end() &&
                    visited.find(curIndex) == visited.end()) {

                    q.push(make_pair(nextIdx, true));
                    visited.insert(curIndex);
                }
            }
        }
        steps++;
    }
    return -1;
}







#endif //DATASTRUCT_ALGORITHM_JUMP_GAME_H

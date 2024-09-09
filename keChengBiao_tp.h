//
// Created by kai.chen on 2022/1/3.
//
//      1. 课程表I
//      2. 课程表II
//
//      3. 会议室：一个人能否参加所有会议
//      4. 会议室II：最少会议室满足所有会议安排
//      5. 最多可以参加的会议数目
#ifndef DATASTRUCT_ALGORITHM_KECHENGBIAO_TP_H
#define DATASTRUCT_ALGORITHM_KECHENGBIAO_TP_H
#include <algorithm>
using namespace std;


// 1. 课程表I (是否循环依赖)
bool hasCycle = false;
vector<int> postOrder; // 后序遍历结果
// 从节点 s 开始 DFS 遍历，将遍历过的节点标记为 true
void traverse(vector<list<int>>& graph, int s, vector<bool>& onPath, vector<bool>& visited) {
    if(onPath[s]){ // 出现环
        hasCycle = true;
        return;
    }
    // 如果之前访问过，说明该点出现在其他路径上，并且没有循环依赖
    if(visited[s] || hasCycle) return ;

    visited[s] = true;
    onPath[s] = true;
    for (int t : graph[s]) {
        traverse(graph, t, onPath, visited);
    }
    postOrder.push_back(s);
    onPath[s] = false; // 回溯onPath
}

bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<list<int>> graph(numCourses); // buildGraph
    for (auto nums : prerequisites) { // [[1,0],[0,1]]
        // 修完课程 edge[1] 才能修课程 edge[0], 在图中添加一条从 1 指向 0 的有向边
        graph[nums[1]].push_back(nums[0]);
    }
    // 记录一次 traverse 递归经过的节点
    vector<bool> onPath(numCourses, false);
    // 记录遍历过的节点，防止走回头路
    vector<bool> visited(numCourses, false);
    for(int i = 0; i < numCourses; i++) { // 遍历图中的所有节点
        traverse(graph, i, onPath, visited);
    }
    // 只要没有循环依赖可以完成所有课程
    return !hasCycle;
}

// 2. 课程表II (求解拓扑排序)
// 思路：DFS 保存后序遍历，倒过来就是拓扑顺序
vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
    if( !canFinish(numCourses, prerequisites)){
        return vector<int>{};
    }
    reverse(postOrder.begin(), postOrder.end());
    return postOrder;
}


void testCanFinish(){
    int courses = 5;
    vector<vector<int>> prerequisites =
            { {3,0},
              {4,1},
              {1,2},
              {2,3},
              {2,4}};
    cout<< canFinish(courses, prerequisites)<<endl;
}

// 3. 会议室：一个人能否参加所有会议
// 给定一个会议时间安排的数组，每个会议时间都会包括开始和结束的时间 [[s1,e1],[s2,e2],…] (si < ei)，请你判断一个人是否能够参加这里面的全部会议。
bool compareAttendMeeting(vector<int> a, vector<int> b){
    return a[0] == b[0]?a[1]>b[1]:a[0]<b[0];
}
bool canAttendMeetings(vector<vector<int>> intervals){
    if(intervals.size() <=1){
        return true;
    }
    // int n = intervals.size();
    //先将会议数组按照开始的时间进行排序
    sort(intervals.begin(),intervals.end(), compareAttendMeeting);
    for(int i=0; i<intervals.size()-1; i++){
        //如果后面一个会议的开始时间小于前面一个会议的结束时间，则此人不能参加全部的会议
        if(intervals[i][1]>intervals[i+1][0]){
            return false;
        }
    }
    return true;
}
// 4. 会议室II：最少会议室满足所有会议安排
//   最大重叠区间重叠次数
int minMeetingRooms(vector<vector<int>> intervals){
    if(intervals.size() <=1){
        return 1;
    }
    //先将会议数组按照开始的时间越小、结束时间越大进行排序
    sort(intervals.begin(),intervals.end(), compareAttendMeeting);
    // pq 存放没结束的会议的结束时间，pq.size时间发生重叠的会议个数
    priority_queue<int> pq;
    int meetingCnt = 0;
    // 遍历 meeting起止时间
    for(const auto & curMeetingInterval : intervals){
        // 当前meeting开始的时间 大于等于 之前meeting的最大结束时间。
        //    将所有已经可以结束的会议pop出去，他们和当前会议不再会发生重叠
        while(!pq.empty() && curMeetingInterval[0] >= pq.top()){
            pq.pop();
        }
        pq.push(curMeetingInterval[1]);
        meetingCnt = max(meetingCnt, int(pq.size()));
    }

    return meetingCnt;
}
void test_minMeetingRooms(){
    vector<vector<int>> meetings = {{0, 30},{5, 10},{15, 20}};
    cout<<minMeetingRooms(meetings)<<endl;
}

// 5. 最多可以参加的会议数目
// 贪心思路：在所有开始时间相同的会议中，我们尽量的去选择结束时间最小的会议，因为结束时间更大的会议的选择天数更多
//      比如在会议：[[1,1],[1,2],[1,3]] 这三个会议中，如果是在第 1 天的话，我们会尽量的选择 [1,1] 这个会议，因为后面的两个会议，
//      分别可以在第 2 天和第 3 天选择，选择的范围更广，只有这样选择，才可以得到能参加更多的会议
//  所以，这里我们需要能快速的选择结束时间最小的会议，而且这个最小的结束时间是动态变化的，因为参加了一个会议，就应该排除这个会议
// 高效的维护动态数据的最小值，我们想到了小顶堆了
// (1)首先，对会议按照开始时间升序排列，排序的目的是为了可以方便的一起选择开始时间相同的会议/ 这里也可以维护一个【开始天】 和 【结束天】的映射
// (2)然后从第 1 天开始，依次判断每一天是否可以参加会议，记当天为 currDay ，从第 1 天开始
// (3)顺序遍历会议，将开始时间等于 currDay 的会议的结束时间放入小顶堆
// (4)将堆顶结束时间小于 currDay 的会议从堆中删除，这些会议都是过时的，参加不到的会议
// (5)如果堆不为空，则选择会议结束时间最小的会议参加，表示 currDay 这一天可以参加会议
// (6)currDay 往后走一天，判断下一天是否可以参加会议

int maxEvents_raw(vector<vector<int>>& events) {
    sort(events.begin(), events.end(), [](const vector<int>& e1, const vector<int>& e2) {
        return e1[1] < e2[1];
    });

    unordered_set<int> res;
    for(vector<int> e: events) {
        for(int d = e[0]; d <= e[1]; d++) {
            if(res.find(d) == res.end()) {
                res.insert(d);
                break;
            }
        }
    }

    return res.size();
} // 纯粹的贪心，不维护堆，时间复杂度n^2

struct cmpatmeeting{
    bool operator()(pair<int,int>&a,pair<int,int>&b){
        return a.second>b.second;
    }
};

int maxEvents1(vector<vector<int>>& events) {
    priority_queue<pair<int, int>, vector<pair<int, int>>, cmpatmeeting> q;
    int n = events.size();

    int res = 0;
    sort(events.begin(), events.end(), [](vector<int> &a, vector<int> &b) {
        return a[0] < b[0];
    });
    int endday = 0;
    for (int i = 0; i < n; i++) {
        endday = max(endday, events[i][1]);
    }
    int i = 0, j = 0;
    while (i <= endday) {
        while (j < n && i >= events[j][0]) {
            q.push({events[j][0], events[j][1]});
            j++;
        }
        while (!q.empty() && q.top().second < i) {
            q.pop();
        }
        if (!q.empty()) {
            q.pop();
            res++;
        }
        i++;
    }
    return res;
}

int maxEvents(vector<vector<int>>& events) {
    int maxDay = 0;
    // 构建一个【开始天】 和 【结束天】的映射
    unordered_map<int, vector<int>> day2days;
    for (vector<int>& event : events){
        maxDay = max(maxDay, event[1]);
        day2days[event[0]].push_back(event[1]);
    }

    // 记录参见会议的次数
    int res = 0;
    // 小顶堆队列
    priority_queue<int, vector<int>, greater<int>> q;
    for (int i = 1; i <= maxDay; ++i){
        // 增加新的结束时间
        if (day2days.find(i) != day2days.end()){
            for (int day : day2days[i]){
                q.push(day);
            }
        }
        // 删除队列里结束时间小于i的会议：因为它们已经结束了，无法再选择
        while (!q.empty() && q.top() < i){
            q.pop();
        }

        // 直接取最小结束时间会议，次数+1
        if (!q.empty()){
            q.pop();
            ++res;
        }
    }
    return res;
} // 时间复杂度: O(S * logn) S是结束时间的上界，


#endif //DATASTRUCT_ALGORITHM_KECHENGBIAO_TP_H

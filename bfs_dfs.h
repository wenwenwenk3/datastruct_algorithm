//
// Created by kai.chen on 2021/12/20.
//
//      1. 二叉树的高度
//      2. 二叉树层序遍历
//      3. 锯齿形层次遍历
//      4. 机器人的运动范围
//
//      5. 打开转盘锁
//
//      d1. 二叉树路径总和等于target的路径
//        d1.1 二叉树的所有距离为k的节点
//        d1.2 求根到叶子节点数字之和
//        d1.3 二叉树的所有路径
//        d1.4 二叉树中的最大路径和
//        d1.5 输出二叉树
//
//      d2. 二叉树的直径
//        d2.1 二叉树的最大宽度
//      d3. 平衡二叉树的判断
//
//      d4. 另一个树的子树判断
//      d5. 解数独
//      d6. 网格中的单词搜索

//      b1. 二分图的判断
//
//      b2. 二叉树的堂兄弟节点
//
#ifndef DATASTRUCT_ALGORITHM_BFS_DFS_H
#define DATASTRUCT_ALGORITHM_BFS_DFS_H
#include <queue>
#include "binary_tree_normal_op.h"

// 1. 二叉树的最小高度
int minDepth(TreeNode* root){
    if(root == nullptr) return 0;
    queue<TreeNode*> q;
    q.push(root);
    int depth = 1;
    while(!q.empty()){
        int sz = q.size();
        for(int i = 0; i < sz; i++){
            TreeNode* cur = q.front();
            q.pop();
            if(cur->left == nullptr && cur->right == nullptr){
                return depth;
            }
            if(cur->left != nullptr) q.push(cur->left);
            if(cur->right != nullptr) q.push(cur->right);
        }
        depth++;
    }
    return depth;
}

// 2. 二叉树的层序遍历
vector<vector<int>> LevelOrder(TreeNode* root){
    vector<vector<int>> res;
    if(root == nullptr) return res;
    queue<TreeNode*> q;
    q.push(root);
    while(!q.empty()){
        int sz = q.size();
        vector<int> vec;
        for(int i = 0; i < sz; i++){
            TreeNode* cur = q.front();
            q.pop();
            vec.push_back(cur->val);
            if(cur->left != nullptr) q.push(cur->left);
            if(cur->right != nullptr) q.push(cur->right);
        }
        res.push_back(vec);
    }
    return res;
}

// 2.1 右视图
vector<int> rightView(TreeNode* root){
    vector<int> res;
    if(root == nullptr) return res;
    queue<TreeNode*> q;
    q.push(root);
    while(!q.empty()){
        int sz = q.size();
        vector<int> vec;
        for(int i = 0; i < sz; i++){
            TreeNode* cur = q.front();
            q.pop();
            vec.push_back(cur->val);
            if(cur->left != nullptr) q.push(cur->left);
            if(cur->right != nullptr) q.push(cur->right);
            if(i == sz-1) res.push_back(cur->val);
        }
    }
    return res;
}

// 3. 锯齿形层次遍历
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    vector<vector<int>> res;
    if(root == nullptr){
        return res;
    }
    queue<TreeNode*> q;
    q.push(root);
    bool left_order = true;
    while(!q.empty()){
        deque<int> dq;
        int sz = q.size();
        for(int i =0; i < sz; i++){
            TreeNode* cur = q.front();
            q.pop();
            if(left_order){
                dq.push_back(cur->val);
            }else{
                dq.push_front(cur->val);
            }
            if(cur->left != nullptr) q.push(cur->left);
            if(cur->right != nullptr) q.push(cur->right);
        }
        left_order = !left_order;
        res.emplace_back(dq.begin(), dq.end());
    }
    return res;
}


// 4. 机器人的运动范围
// 计算 x 的数位之和
int cal(int x) {
    int res=0;
    for (; x; x /= 10) {
        res += x % 10;
    }
    return res;
}
int movingCount(int m, int n, int k) {
    if (!k) return 1;
    queue<pair<int,int> > q;

    vector<vector<bool>> memo(m, vector<bool>(n, false));
    q.push(make_pair(0, 0));

    memo[0][0] = true;
    int count = 1;
    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();
        // 做选择，向下或向右方向
        if(x+1 < m && !memo[x+1][y] && cal(x+1)+cal(y) <= k){
            q.push(make_pair(x+1, y));
            memo[x+1][y] = true;
            count++;
        }
        if(y+1 < n && !memo[x][y+1] && cal(x)+cal(y+1) <= k){
            q.push(make_pair(x, y+1));
            memo[x][y+1] = true;
            count++;
        }
    }
    return count;
} // 时间复杂度O(m*n), 最差的时候所有的格子都能进入 其中每个格子最多被访问2次
//  空间复杂度O(m*n),需要标记结构用来标记每个格子是否被走过


// 5. 打开转盘锁
// 题目描述：找出从初始数字 0000 到解锁数字 target 的最小旋转次数。每次旋转都只能旋转一个拨轮的一位数字
// 思路：BFS搜索，每一个状态，每一位都可以向前向后转
//
// 时间复杂度：O(b^d * d^2 + md)，其中 b 是数字的进制，d 是转盘数字的位数，m 是数组 deadends 的长度，在本题中 b=10，d=4。
//  状态总数：10^4，对于每一个状态需要O(d)的时间旋转
char prevNum(char x){
    if(x == '0') return '9';
    else return x-1;
}
char nextNum(char x){
    if(x == '9') return '0';
    else return x+1;
}

// 计算s通过一次旋转的所有下一个可能状态
vector<string> nextStatus(string s){
    vector<string> res;
    for (int i = 0; i < 4; ++i) {
        // 做选择，每一位 都可以 向前或 向后旋转
        char num = s[i];
        s[i] = prevNum(num);
        res.push_back(s);
        s[i] = nextNum(num);
        res.push_back(s);

        s[i] = num; // 选择完，改回当前位置选择前
    }
    return res;
}

int openLock(vector<string>& deadends, string& target) {
    if (target == "0000") {
        return 0;
    }

    unordered_set<string> dead(deadends.begin(), deadends.end());
    if (dead.count("0000")) return -1;

    // first: s  second: steps
    queue<pair<string, int>> q;
    q.push(make_pair("0000", 0));
    unordered_set<string> visited = {"0000"};

    while (!q.empty()) {
        auto cur = q.front();
        string cur_status = cur.first;
        int cur_steps = cur.second;

        q.pop();
        // 做选择
        vector<string> nxtStatus = nextStatus(cur_status);

        for (auto status : nxtStatus) {
            // 只有当当前状态是合法的，才可以继续往下转
            if (!visited.count(status) && !dead.count(status)) {
                if (status == target) {
                    return cur_steps + 1;
                }
                q.push(make_pair(status, cur_steps + 1));
                visited.insert(move(status));
            }
        }
    }
    return -1;
} // 时间复杂度： 这是个bfs广度搜索，最差会尝试所有可能 10^4
// 空间复杂度：队列里最多存放 10^4 个元素，每个元素长度为长4位的string



// d1. 二叉树路径总和等于target
// 题目描述：找出所有满足从根节点到叶子结点路径总和等于target的路径
// 思路： dfs搜索，退出条件是当前节点到了叶子节点且path路径上的和刚好等于target
vector<vector<int>> res;
vector<int> path;
void dfs(TreeNode* root, int target) {
    if (root == nullptr) {
        return;
    }
    path.push_back(root->val);
    if (root->left == nullptr && root->right == nullptr && target == root->val) {
        res.push_back(path);
    }
    dfs(root->left, target - root->val);
    dfs(root->right, target - root->val);
    path.pop_back();
}

vector<vector<int>> pathSum(TreeNode* root, int target) {
    dfs(root, target);
    return res;
}

// d1延伸. 路径总和III,路径可以是任意一段，不要求从根到叶子结点
// 求路径总和等于target的路径 的数目
// 树上前缀和 + 哈希
unordered_map<long long, int> prefix;
int dfs(TreeNode *root, long long curr, int targetSum) {
    if (root == nullptr) return 0;

    int ret = 0;
    curr += root->val;
    if (prefix.count(curr - targetSum)) {
        ret = prefix[curr - targetSum];
    }

    prefix[curr]++;
    ret += dfs(root->left, curr, targetSum);
    ret += dfs(root->right, curr, targetSum);
    prefix[curr]--;

    return ret;
}

int pathSumIII(TreeNode* root, int targetSum) {
    prefix[0] = 1;
    return dfs(root, 0, targetSum);
} // 时间复杂度 O(n), 空间复杂度O(n)


// d1.1 二叉树的所有距离为k的节点
// 题目描述：返回到目标结点 target 距离为 k 的所有结点的值的列表。 答案可以以 任何顺序 返回。
// 思路：DFS + 哈希表
class Solution {
    // 哈希表用来快速找到每个节点的父节点
    unordered_map<int, TreeNode*> parents;
    vector<int> ans;

    void findParents(TreeNode* node) {
        if (node->left != nullptr) {
            parents[node->left->val] = node;
            findParents(node->left);
        }
        if (node->right != nullptr) {
            parents[node->right->val] = node;
            findParents(node->right);
        }
    }

    void findAns(TreeNode* node, TreeNode* from, int depth, int k) {
        if (node == nullptr) {
            return;
        }
        if (depth == k) {
            ans.push_back(node->val);
            return;
        }
        if (node->left != from) {
            findAns(node->left, node, depth + 1, k);
        }
        if (node->right != from) {
            findAns(node->right, node, depth + 1, k);
        }
        if (parents[node->val] != from) {
            findAns(parents[node->val], node, depth + 1, k);
        }
    }

public:
    vector<int> distanceK(TreeNode* root, TreeNode* target, int k) {
        // 从 root 出发 DFS，记录每个结点的父结点
        findParents(root);

        // 从 target 出发 DFS，寻找所有深度为 k 的结点
        findAns(target, nullptr, 0, k);

        return ans;
    }
};

//  建图 + dfs
//深度优先遍历图
vector<int> res;
void dfs(vector<vector<int>>& graph,int begin, int k,vector<int> &visited){
    if(k == 0 ){
        res.push_back(begin);
    }
    visited[begin] = 1;
    for(int i = 0; i < graph[begin].size(); ++i){
        int neighbor = graph[begin][i];
        if(visited[neighbor] == 0){
            dfs(graph,neighbor,k - 1,visited);
        }
    }
    return;
}
//将二叉树转为无向图
void buildGraph(TreeNode* root,vector<vector<int>>& graph){
    if(!root) return;
    if(root->left){
        graph[root->val].push_back(root->left->val);
        graph[root->left->val].push_back(root->val);
        buildGraph(root->left,graph);
    }
    if(root->right){
        graph[root->val].push_back(root->right->val);
        graph[root->right->val].push_back(root->val);
        buildGraph(root->right,graph);
    }
    return;
}
vector<int> distanceK(TreeNode* root, TreeNode* target, int k){
    vector<vector<int>> graph(MAX_SIZE);//邻接表法表示图
    vector<int> visited(MAX_SIZE,0); //对图 dfs 时， 标记访问过的节点
    buildGraph(root,graph);//从二叉树建无向图
    res.clear();
    dfs(graph,target->val,k,visited);//求图上 离节点 target 距离为 k 的所有节点
    return res;
}


// d1.2 求根到叶子节点数字之和
int dfs(TreeNode* root, int sum) {
    if (root == nullptr) {
        return 0;
    }
    int newsum = sum * 10 + root->val;
    if (root->left == nullptr && root->right == nullptr) {
        return newsum;
    }
    else {
        return dfs(root->left, newsum) + dfs(root->right, newsum);
    }
}
int sumNumbers(TreeNode* root) {
    return dfs(root, 0);
}// 时间复杂度：O(n)，其中 n 是二叉树的节点个数。对每个节点访问一次。
// 空间复杂度：O(n)，递归调用的栈空间最差情况


// d1.3 二叉树的所有路径
vector<string> res;
void dfs(TreeNode* root, string path) {
    if (root == nullptr) return ;
    path += to_string(root->val);
    if (root->left == nullptr && root->right == nullptr) {  // 当前节点是叶子节点
        res.push_back(path); // 把路径加入到答案中
        return ;
    }
    path += "->";  // 当前节点不是叶子节点，继续递归遍历
    dfs(root->left, path);
    dfs(root->right, path);
}

vector<string> binaryTreePaths(TreeNode* root) {
    dfs(root, "");
    return res;
}


// d1.4 二叉树中的最大路径和



// d1.5 输出二叉数
// root = [1,2,3,null,4]
//输出：
//[["","","","1","","",""],
//["","2","","","","3",""],
//["","","4","","","",""]]
class printTreeSolution {
public:
    int n;

    int dfs_depth(TreeNode* u, int depth){
        int left_depth = depth, right_depth = depth;
        if (u -> left != nullptr) left_depth = dfs_depth(u -> left, depth + 1);
        if (u -> right != nullptr) right_depth = dfs_depth(u -> right, depth + 1);
        return max(left_depth, right_depth);
    }

    void build(TreeNode* u, vector<vector<string>>& mat, int depth, int loc){
        mat[depth][loc] = to_string(u -> val);
        if (u -> left != nullptr) build(u -> left, mat, depth + 1, loc - (1 << (n - depth - 2)));
        if (u -> right != nullptr) build(u -> right, mat, depth + 1, loc + (1 << (n - depth - 2)));
    }

    vector<vector<string>> printTree(TreeNode* root) {
        n = dfs_depth(root, 1);
        vector<vector<string>> res(n, vector<string>((1 << n) - 1, ""));
        build(root, res, 0, ((1 << n) - 1) >> 1);

        return res;
    }
};

// d2. 二叉树的直径
// 题目描述：所谓二叉树的直径是二叉树中最长的路径
// 思路：所以求直径（即求路径长度的最大值）等效于求路径经过节点数的最大值减一。
//    所以问题转化为：后序遍历求路径经过的节点数最大值
//      (有点类似于二叉树的最大路径和见dp_1d)
int resmaxbilength = INT_MIN;
int depthdfs(TreeNode* root){
    if (root == nullptr) {
        return 0; // 访问到空节点了，返回0
    }
    int leftMaxD = depthdfs(root->left); // 左儿子为根的子树的深度
    int rightMaxD = depthdfs(root->right); // 右儿子为根的子树的深度
    resmaxbilength = max(resmaxbilength, leftMaxD + rightMaxD + 1); // 并更新ans
    return max(leftMaxD, rightMaxD) + 1; // 返回该节点为根的子树的深度
}
int diameterOfBinaryTree(TreeNode* root) {
    resmaxbilength = 1;
    depthdfs(root);
    return resmaxbilength-1;
}

// d2.1 二叉树的宽度
// 题目描述：每一层的宽度被定义为两个端点（该层最左和最右的非空节点，两端点间的null节点也计入长度）之间的长度。
// 思路：bfs，队列元素使用pair，多用一个int值记录当前层的索引
// 左子树是父节点的index * 2,右子树是 index * 2 + 1
#define LL long long
int widthOfBinaryTree(TreeNode* root) {
    if(root == nullptr)    return 0;
    queue<pair<TreeNode*, LL>> q;  //pair的第二个位置记录当前是第几个节点
    q.push({root, 1});
    LL width = 0;
    while(!q.empty()){
        //start是本层起点, index是本层当前遍历到的节点的索引
        int sz = q.size();
        LL start = q.front().second, index;
        for(int i = 0; i < sz; i++){
            TreeNode *cur = q.front().first;
            index = q.front().second;
            q.pop();
            if(cur->left != nullptr)   {
                q.push({cur->left , index*2 - start*2});  //防止索引位置太大溢出
            }
            if(cur->right != nullptr)  {
                q.push({cur->right, index*2 + 1 - start*2});
            }
        }
        width = max(width, index - start + 1);
    }
    return width;
}

// d3. 平衡二叉树的判断
// 题目描述：平衡的二叉树左右子树的高度相差不超过1
// 思路：递归
int height(TreeNode* root) {
    if (root == nullptr) {
        return 0;
    }
    int leftHeight = height(root->left);
    int rightHeight = height(root->right);
    if (leftHeight == -1 || rightHeight == -1 || abs(leftHeight - rightHeight) > 1) {
        return -1;
    } else {
        return max(leftHeight, rightHeight) + 1;
    }
}

bool isBalanced(TreeNode* root) {
    return height(root) != -1;
}

// d4. 另一个树的子树判断
bool check(TreeNode *t1, TreeNode *t2) {
    if(t1 == nullptr && t2 == nullptr) return true;
    if(t1 == nullptr || t2 == nullptr) return false;
    if(t1->val != t2->val) return false;

    return check(t1->left, t2->left) && check(t1->right, t2->right);
}

bool dfs(TreeNode *t1, TreeNode *t2) {
    if (t1 == nullptr) return false;
    return check(t1, t2) || dfs(t1->left, t2) || dfs(t1->right, t2);
}

bool isSubtree(TreeNode *s, TreeNode *t) { // 判断t是否是s的子树
    return dfs(s, t);
}



// d5. 解数独
// 前菜，判断当前板子上的数独有效
bool isValidSudoku(vector<vector<char>>& board) {
    int rows[9][9];
    int columns[9][9];
    int subboxes[3][3][9];

    memset(rows,0,sizeof(rows));
    memset(columns,0,sizeof(columns));
    memset(subboxes,0,sizeof(subboxes));
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            char c = board[i][j];
            if (c != '.') {
                int index = c - '0' - 1;
                rows[i][index]++;
                columns[j][index]++;
                subboxes[i / 3][j / 3][index]++;
                if (rows[i][index] > 1 || columns[j][index] > 1 || subboxes[i / 3][j / 3][index] > 1) {
                    return false;
                }
            }
        }
    }
    return true;
}
// 回溯，挨个开始填
class Solution {
private:
    //三个布尔数组 表明 行, 列, 还有 3*3 的方格的数字是否被使用过
    bool mCol[9][9];
    bool mRow[9][9];
    bool mSquared[9][9]; // [i][j] i是方格编号row/3 * 3 + col/3，j数字存在与否。
    vector<pair<int, int>> mSpace;
public:
    void solveSudoku(vector<vector<char>>& board) {
        for (auto i = 0; i < board.size(); ++i) {
            for (auto j = 0; j < board[i].size(); ++j) {
                char c = board[i][j];
                if (c == '.') {
                    mSpace.emplace_back(i, j);
                }
                else {
                    mRow[i][c - '1'] = true; // 标记该行数字c已使用
                    mCol[j][c - '1'] = true;
                    mSquared[i / 3 * 3 + j / 3][c - '1'] = true;
                }
            }
        }
        Backtrack(board, 0); // 填空格，从第一个开始填
    }

    bool Backtrack(vector<vector<char>>& board, int index) {
        if (index == mSpace.size()) {
            return true;
        }

        int row = mSpace[index].first;
        int col = mSpace[index].second;
        // 回溯，挨个尝试
        for (auto c = '1'; c <= '9'; ++c) {
            int num = c - '1';

            // 保证 行不重复 列不重复 小方格不重复
            if (!mRow[row][num] && !mCol[col][num] && !mSquared[row / 3 * 3 + col / 3][num]) { // 方格编号row / 3 * 3 + col / 3
                board[row][col] = c;
                mRow[row][num] = true;
                mCol[col][num] = true;
                mSquared[row / 3 * 3 + col / 3][num] = true;

                if (Backtrack(board, index + 1)) { // 递归，如果有正确结果返回true
                    return true;
                }

                mRow[row][num] = false;
                mCol[col][num] = false;
                mSquared[row / 3 * 3 + col / 3][num] = false;
            }
        }
        return false;
    }
};


// d6. 网格中的单词搜索
// 题目描述：m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
bool exist(vector<vector<char>>& board, string& word) {
    for (int i = 0; i < board.size(); ++i) {
        for (int j = 0; j < board[i].size(); ++j) {
            if (board[i][j] == word[0]) {
                // 从i,j开始搜索
                if (dfs(board, word, i, j, 0)) return true;
            }
        }
    }
    return false;
}
bool dfs(vector<vector<char>>& board, string& word, int i, int j, int l) {
    if (l == word.size()) { // l是当前长度, 退出条件
        return true;
    }
    if (i < 0 || j < 0 || i >= board.size() || j >= board[i].size()) {
        return false;
    }

    if (board[i][j] != word[l]) {
        return false;
    }
    char t = board[i][j];
    board[i][j] = '0'; // 标记访问过
    bool res = dfs(board, word, i + 1, j, l + 1) ||
               dfs(board, word, i - 1, j, l + 1) ||
               dfs(board, word, i, j + 1, l + 1) ||
               dfs(board, word, i, j - 1, l + 1);
    board[i][j] = t; // backtrack
    return res;
}



// b1. 二分图判定
// 思路：BFS搜索，记录颜色和访问与否，当访问过并且颜色相同 则不满足
// 记录图是否符合二分图性质
bool ok = true;
// 记录图中节点的颜色，false 和 true 代表两种不同颜色
vector<bool> color;
// 记录图中节点是否被访问过
vector<bool> visited;

bool isBipartite(vector<vector<int>>& graph) {
    int n = graph.size();
    color.resize(n);
    visited.resize(n);

    for (int v = 0; v < n; v++) {
        if (!visited[v]) {
            bfs(graph, v);
        }
    }
    return ok;
}
// 从 start 节点开始进行 BFS 遍历
void bfs(vector<vector<int>>& graph, int start) {
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty() && ok) {
        int cur = q.front();
        q.pop();
        // 从节点 cur 向所有相邻节点扩散
        for (int w : graph[cur]) {
            if (!visited[w]) { // 相邻节点 w 没有被访问过
                // 那么应该给节点 w 涂上和节点 cur 不同的颜色
                color[w] = !color[cur];
                // 标记 w 节点，并放入队列
                visited[w] = true;
                q.push(w);
            }
            else { // 相邻节点 w 已经被访问过
                // 且 w 和 cur 的颜色相同，则此图不是二分图
                if (color[w] == color[cur]) {
                    ok = false;
                }
            }
        }
    }
}


// b2. 二叉树的堂兄弟节点
// 题目描述：如果二叉树的两个节点深度相同，但 父节点不同 ，则它们是一对堂兄弟节点。
//      只有与值 x 和 y 对应的节点是堂兄弟节点时，才返回 true 。否则，返回 false
//   限制：二叉树的节点数介于 2 到 100 之间。
//        每个节点的值都是唯一的、范围为 1 到 100 的整数
// 简化目标为：求某个节点的「父节点」&「所在深度」

// dfs解法，不存储father节点
int dfsCousins(TreeNode *root, TreeNode *&father, int x) {
    if (root == nullptr) return -1;
    if (root->val == x) {
        return 1;
    }

    father = root;
    int l = dfsCousins(root->left, father, x);
    if (l != -1) return l + 1; // 在左子树找到了x，返回对应深度

    father = root;
    int r = dfsCousins(root->right, father, x);
    if (r != -1) return r + 1; // 在右子树找到了x，返回对应深度

    return -1; // 否则返回-1
}

bool isCousinsDFS(TreeNode* root, int x, int y) {
    int d1, d2;
    TreeNode *f1 = nullptr, *f2 = nullptr;
    d1 = dfsCousins(root, f1, x);
    d2 = dfsCousins(root, f2, y);
    return d1 == d2 && f1 != f2;
}

// bfs 解法 不存储父节点的解法
bool isCousinsBFS(TreeNode* root, int x, int y) {
    TreeNode *xf=nullptr, *yf=nullptr, *cur=nullptr;
    queue<TreeNode*> q;
    q.push(root);
    while(!q.empty()){
        int sz = q.size();
        //遍历当前层，同时出现且异父，返回真；均未出现则继续遍历下一层；否则返回假
        for(int i=0; i < sz; i++){
            cur=q.front();
            if(cur->left){
                q.push(cur->left);
                xf=cur->left->val==x?cur:xf;
                yf=cur->left->val==y?cur:yf;
            }
            if(cur->right){
                q.push(cur->right);
                xf=cur->right->val==x?cur:xf;
                yf=cur->right->val==y?cur:yf;
            }
            q.pop();
        }
        if(xf&&yf&&(xf!=yf)) return true;
        if(xf||yf) return false;
    }
    //啥都没找到，返回假
    return false;
}

// BFS 存储父节点的解法
struct data{
    int depth;
    TreeNode *father, *root;
};
bool isCousinsBFS2(TreeNode* root, int x, int y) {
    int d1, d2;
    TreeNode *f1 = nullptr, *f2 = nullptr;
    queue<data> q;
    q.push({0, root, nullptr});
    while (!q.empty()) {
        data temp = q.front();
        q.pop();
        if (temp.root->val == x) {d1 = temp.depth, f1 = temp.father;}
        if (temp.root->val == y) {d2 = temp.depth, f2 = temp.father;}
        if (temp.root->left) {
            q.push({temp.depth + 1, temp.root->left, temp.root});
        }
        if (temp.root->right) {
            q.push({temp.depth + 1, temp.root->right, temp.root});
        }
    }
    return d1 == d2 && f1 != f2;
}


#endif //DATASTRUCT_ALGORITHM_BFS_DFS_H


#if 0
// 输入起点，进行 BFS 搜索
int BFS(Node start) {
    Queue<Node> q; // 核心数据结构
    Set<Node> visited; // 避免走回头路

    q.offer(start); // 将起点加入队列
    visited.add(start);

    int step = 0; // 记录搜索的步数
    while (q not empty) {
        int sz = q.size();
        /* 将当前队列中的所有节点向四周扩散一步 */
        for (int i = 0; i < sz; i++) {
            Node cur = q.poll();
            printf("从 %s 到 %s 的最短距离是 %s", start, cur, step);

            /* 将 cur 的相邻节点加入队列 */
            for (Node x : cur.adj()) {
                if (x not in visited) {
                    q.offer(x);
                    visited.add(x);
                }
            }
        }
        step++;
    }
}
注意，我们的 BFS 算法框架也是while循环嵌套for循环的形式，也用了一个step变量记录for循环执行的次数，
无非就是多用了一个visited集合记录走过的节点，防止走回头路罢了。
#endif

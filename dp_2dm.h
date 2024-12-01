//
// Created by kai.chen on 2021/12/21.
//
// 二维dp 显式二维矩阵题    解决思路： DFS 或 并查集思路
//      1.小兵走棋盘的不同路径
//          1.1 不同路径II 含障碍物
//          1.2 不同的路径III  （见文末注释部分）
//              1.2.1 followup 获取钥匙的最短路径 （LC864）
//          1.3. 最小路径和
//          1.4. 礼物的最大价值(最大路径和)
//
//      2.机器人的运动范围
//
//      3.岛屿数量
//       3.1 被围绕的区域(替换XO, 即岛屿数量II)
//       3.2 封闭岛屿的数量

//      4. 岛屿的最大面积
//       4.1 封闭岛屿的面积总和 (飞地数量)
//       4.2 岛屿的最大矩形面积
//       4.3 岛屿的最大正方形面积
//
//      5. 边界着色
//
//      6. 子岛屿的数量
//      7. 不同的岛屿数量
//
#ifndef DATASTRUCT_ALGORITHM_DP_2D_M_H
#define DATASTRUCT_ALGORITHM_DP_2D_M_H
#include <vector>
#include <list>
using namespace std;

// 1.小兵走棋盘的不同路径
// 题目描述： m*n的棋盘，小兵每次只能向下/向右走一步，问从左上角到右下角有多少不同路径
// 思路： dp(i, j)表示从左上角走到 (i,j) 的路径数量，其中i=[0, m), j=[0, n)
//    dp[i][j] = dp[i-1][j] + dp[i][j-1]
//    base case: dp[0][j] = 1, dp[i][0] = 1
int uniquePaths(int m, int n) {
    vector<vector<int>> dp(m, vector<int>(n));
    // base case: dp[0][j] = 1, dp[i][0] = 1
    for (int i = 0; i < m; ++i) {
        dp[i][0] = 1;
    }
    for (int j = 0; j < n; ++j) {
        dp[0][j] = 1;
    }
    // dp[i][j] = dp[i-1][j] + dp[i][j-1]
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
} // 时间复杂度O(n*m)

// 1.1 不同路径II 含障碍物
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    int m = obstacleGrid.size(), n = obstacleGrid.at(0).size();
    vector<vector<int>> dp(m, vector<int>(n, 0));
    dp[0][0] = obstacleGrid[0][0] == 1 ? 0 : 1;
    // base case: dp[0][j] = 1, dp[i][0] = 1
    for (int i = 1; i < m; ++i) {
        if(obstacleGrid[i][0] == 1) dp[i][0] = 0;
        else dp[i][0] = dp[i-1][0];
    }
    for (int j = 1; j < n; ++j) {
        if(obstacleGrid[0][j] == 1) dp[0][j] = 0;
        else dp[0][j] = dp[0][j-1];
    }
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (obstacleGrid[i][j] == 1) {
                dp[i][j] = 0;
                continue;
            }
            if (obstacleGrid[i][j] == 0) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
    }

    return dp[m - 1][n - 1];
}

// 变体1.1 最小路径和
//  题目描述：在棋盘路径上有一个带权重的grid，问从左上角到右下角的最小路径和
//  思路：dp[i][j] 表示从左上角出发到 (i,j) 位置的最小路径和
//    显然：dp[i][j] = min(dp[i−1][j], dp[i][j−1])
//     base case: dp[0][0] = grid[0][0]
//        dp[i][0]=dp[i−1][0]+grid[i][0]， dp[0][j]=dp[0][j−1]+grid[0][j]
int minPathSum(vector<vector<int>>& grid) {
    if (grid.size() == 0 || grid[0].size() == 0) {
        return 0;
    }
    int m = grid.size(), n = grid[0].size(); // m行 n列
    auto dp = vector<vector<int>> (m, vector<int>(n));
    // base case
    dp[0][0] = grid[0][0];
    for (int i = 1; i < m; i++) {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }
    for (int j = 1; j < n; j++) {
        dp[0][j] = dp[0][j - 1] + grid[0][j];
    }
    // dp
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
        }
    }
    return dp[m - 1][n - 1];
}
// 变体1.2 礼物的最大价值
// 类似于 最大路径和，直接套用上面的模版，改成max就完了, 时间复杂度O(M*N)；动态规划需遍历整个 gridgrid 矩阵，使用 O(MN)O(MN) 时间。


// 2.机器人的运动范围
// 题目描述： m*n的方格，限制机器人不能进入行坐标和列坐标的数位之和大于k的格子。问机器人能够到达多少个格子？
//          例如：当k=18时，能进入坐标(35，37)，因为3+5+3+7=18，但不能进入(35,38)
//    思路：BFS算法，每次进入坐标(i,j)判断是否可以进入，如果可以再继续判断相邻的四个格子/其实是相邻的两个格子
// 代码见 bfs_dfs 4.



// 3.岛屿数量
// 题目描述：一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
// 思路：
//      DFS标记思路：每次遇到1，就把周围所有的1感染成2 （FloodFill算法）
//      先说正确的思路，再聊有趣的思路是dp思路、 但考虑不了这种情况{{'1','1','1'},
////                         {'0','1','0'},
////                         {'1','1','1'}};
// 岛屿为m，从i,j开始感染
void infect(vector<vector<char>>& m, int i, int j){
    int row = m.size(), col = m[0].size();
    if(i <0 || i>= row || j < 0 || j >= col || m[i][j] != '1'){ // 注意如果下面那行用0填充的话 那么这里条件应为m[i][j] == '0'
        return ;
    }
    m[i][j] = '2';
    infect(m, i-1, j); // 上
    infect(m, i+1, j); // 下
    infect(m, i, j-1); // 左
    infect(m, i, j+1); // 右
}
int numIslands(vector<vector<char>>& grid) {
    int row = grid.size(), col = grid[0].size();
    int island = 0;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(grid[i][j] == '1'){
                // 每发现一个岛屿，数量+1
                island++;
                // 然后dfs 将岛屿淹了，主要是为了省事，避免维护visited数组。
                infect(grid, i,j);
            }
        }
    }
    return island;
}

void testNumIslands(){
    vector<vector<char>> grid = {{'1','1','1'},
                                 {'0','1','0'},
                                 {'1','1','1'}};
    cout<<numIslands(grid)<<endl;
}


// 3.1 被围绕的区域(岛屿填充) ，
//      X能把O围了，但任何边界上的 O 都不会被填充为 X
//     时间复杂度：O(m * n) 每一个点至多只会被标记一次。
// 思路： DFS搜索
//      for循环遍历四条边的O，我们以它为起点，标记所有与它直接或间接相连的字母 O为'#'。这些肯定是不会被围的
//      左上到右下遍历矩阵，对于每一个字母：
//          如果被标记过'#'，将它还原为O
//          如果没有被标记过的O，则表示被X包围。将它改成X
//      解法2，union-find思路，见进阶
int n, m;
void dfs_sharp(vector<vector<char>>& board, int x, int y) {
    if (x < 0 || x >= n || y < 0 || y >= m || board[x][y] != 'O') {
        return;
    }
    board[x][y] = '#';
    dfs_sharp(board, x + 1, y);
    dfs_sharp(board, x - 1, y);
    dfs_sharp(board, x, y + 1);
    dfs_sharp(board, x, y - 1);
}

void solve(vector<vector<char>>& board) {
    if (board.empty()) return;
    n = board.size();
    m = board[0].size();
    // 把靠边的岛屿淹掉
    // 把靠第一行和最后一行的O变成#
    for (int i = 0; i < n; i++) {
        dfs_sharp(board, i, 0);
        dfs_sharp(board, i, m - 1);
    }
    // 把第一列和最后一列的O变成#
    for (int i = 0; i < m - 1; i++) {
        dfs_sharp(board, 0, i);
        dfs_sharp(board, n - 1, i);
    }
    // 剩下的O都是应该替换的，深下的#都是应该恢复的
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (board[i][j] == 'O') {
                board[i][j] = 'X';
            }
        }
    }
    // 剩下的#都是应该复原的
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (board[i][j] == '#') {
                board[i][j] = 'O';
            }
        }
    }
}

// 3.2 封闭岛屿的数量
//  去掉上一题中的那些靠边的岛屿，剩下的就是封闭岛屿
// 思路：FloodFill思路 或 UnionFind思路
int closedIsland(vector<vector<char>>& grid){
    int row = grid.size(), col = grid[0].size();
    // 把靠边的岛屿淹掉（此处省略，和上题一样）
    // 统计岛屿个数
    int island = 0;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(grid[i][j] == '1'){
                // 每发现一个岛屿，数量+1
                island++;
                // 然后dfs 将岛屿淹了，主要是为了省事，避免维护visited数组。
                infect(grid, i,j);
            }
        }
    }
    return island;
}



// 4. 岛屿的最大面积
// dfs 记录面积值
int dfs(vector<vector<int>>& grid, int x, int y) {
    // grid[x][y] != 1 返回 0
    if (x < 0 || y < 0 || x == grid.size() || y == grid[0].size() || grid[x][y] != 1) {
        return 0;
    }

    grid[x][y] = 0;
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, 1, 0, -1}; //偏移数组
    int area = 1;

    for (int i = 0; i < 4; i++){
        int x_new = x + dx[i];
        int y_new = y + dy[i];
        area += dfs(grid, x_new, y_new);
    }
    return area;
}
int maxAreaOfIsland(vector<vector<int>>& grid) {
    int ans = 0;
    for (int i = 0; i != grid.size(); ++i) {
        for (int j = 0; j != grid[0].size(); ++j) {
            ans = max(ans, dfs(grid, i, j));
        }
    }
    return ans;
}

// 4.1 飞地的数量
//  求封闭岛屿的面积总和
// 思路也是一样的：先把靠边的陆地淹掉，然后去数陆地数量就行
void dfs_sharp2(vector<vector<int>>& board, int x, int y) {
    if (x < 0 || x >= n || y < 0 || y >= m || board[x][y] != 'O') {
        return;
    }
    board[x][y] = INT_MIN;
    dfs_sharp2(board, x + 1, y);
    dfs_sharp2(board, x - 1, y);
    dfs_sharp2(board, x, y + 1);
    dfs_sharp2(board, x, y - 1);
}
int numEnclaves(vector<vector<int>>& grid){
    int m = grid.size(), n = grid[0].size();
    // 把靠边的岛屿淹掉
    // 把靠第一行和最后一行的O变成INT_MIN
    for (int i = 0; i < n; i++) {
        dfs_sharp2(grid, i, 0);
        dfs_sharp2(grid, i, m - 1);
    }
    // 把第一列和最后一列的O变成INT_MIN
    for (int i = 1; i < m - 1; i++) {
        dfs_sharp2(grid, 0, i);
        dfs_sharp2(grid, n - 1, i);
    }

    int landcount = 0;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(grid[i][j] == 1){
                // 每发现一个岛屿，数量+1
                landcount++;
            }
        }
    }
    return landcount;
}

//  4.2 岛屿的最大矩形面积
// 这题思路较难，暴力的方法是遍历所有可能左上角和右下角复杂度M*N*M*N。可以参考柱状图的最大矩形面积 的单调栈优化
//   对于矩阵中任意一个点，我们枚举以该点为右下角的全 1 矩形。
//   一层一层看不就是 柱状图的最大矩形面积，
int maximalRectangle(vector<vector<char>>& matrix) {
    int m = matrix.size();
    if (m == 0) {
        return 0;
    }
    int n = matrix[0].size();

    // height[i][j] = 位置i,j 上边连续1的数量
    vector<vector<int>> height(m, vector<int>(n, 0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] == '1') {
                height[i][j] = (i == 0 ? 0: height[i-1][j]) + 1;
            }
        }
    }

    int max_area = 0;
    for (int i = 0; i < m; i++) {
        // 对每一行， 每个元素 考虑以该点为右下角的全 1 矩形: 高度为height[i][j], 宽度为左边连续1
        //  lf,rt 分别求解 位置以高度 height[i][j]的左边最小位置和右边最小位置
        vector<int> lf(m, 0), rt(m, 0);
        stack<int> stk;
        for (int j = 0; j < n; j++) {
            while (!stk.empty() && height[i][stk.top()] >= height[i][j]) {
                stk.pop();
            }
            lf[i] = stk.empty() ? -1 : stk.top();
            stk.push(i);
        }
        stk = stack<int>();
        for (int j = n - 1; j >= 0; j--) {
            while (!stk.empty() && height[i][stk.top()] >= height[i][j]) {
                stk.pop();
            }
            rt[i] = stk.empty() ? m : stk.top();
            stk.push(i);
        }

        for (int j = 0; j < n; j++) {
            int wight = lf[j] - rt[j] - 1;
            int area = wight * height[i][j];
            max_area = max(max_area, area);
        }
    }
    return max_area;
}// 时间复杂度：O(mn)，计算 left 矩阵需要 O(mn) 的时间；对每一列应用柱状图算法需要 O(m) 的时间，一共需要 O(mn) 的时间。
// 空间复杂度：O(mn)，分配了一个与给定矩阵等大的数组，用于存储每个元素的左边连续 1 的数量。


// 4.3 岛屿的最大正方形面积
// 定义dp(i,j) 表示以 (i,j) 为右下角，且只包含 1 的正方形的边长最大值
// dp(i,j)=min(dp(i−1,j),dp(i−1,j−1),dp(i,j−1))+1
int maximalSquare(vector<vector<char>>& matrix) {
    if (matrix.size() == 0 || matrix[0].size() == 0) {
        return 0;
    }
    int maxSide = 0;
    int rows = matrix.size(), columns = matrix[0].size();
    vector<vector<int>> dp(rows, vector<int>(columns));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (matrix[i][j] == '1') {
                if (i == 0 || j == 0) {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }
                maxSide = max(maxSide, dp[i][j]);
            }
        }
    }
    int maxSquare = maxSide * maxSide;
    return maxSquare;
} // 时间复杂度：O(mn)，需要遍历原始矩阵中的每个元素计算 dp 的值。
// 空间复杂度：O(mn)，创建了一个和原始矩阵大小相同的矩阵 dp


// 5. 边界着色
// BFS：用一个队列来实现广度优先搜索遍历连通分量，并用一个大小和 grid 相同的矩阵 visited 来记录当前节点是否被访问过，
//       BFS过程中记录边界点存入数组 borders 中
vector<vector<int>> colorBorder(vector<vector<int>>& grid, int row, int col, int color) {
    int m = grid.size(), n = grid[0].size();

    vector<vector<bool>> visited(m, vector<bool>(n, false));
    vector<pair<int, int>> borders;
    vector<vector<int>> dirs = {{0,1}, {1,0}, {0,-1}, {-1,0}}; // 方向数组，上右下左的坐标偏移量

    queue<pair<int, int>> q;
    q.push(make_pair(row, col));

    while (!q.empty()) {
        auto cur = q.front();
        q.pop();
        int x = cur.first, y = cur.second;
        bool isBorder = false;
        for (auto dir : dirs) {
            int xi = x + dir[0], yi = y + dir[1];
            if (!(xi >= 0 && xi < m && yi >= 0 && yi < n && grid[xi][yi] == grid[row][col])) {
                isBorder = true;
            } else if(!visited[xi][yi]){
                visited[xi][yi] = true;
                q.push(make_pair(xi, yi));
            }
            if(isBorder) {
                borders.emplace_back(x, y);
            }
        }
    }

    for (auto& point : borders) {
        grid[point.first][point.second] = color;
    }
    return grid;
}

// 思路2：DFS: 用递归来实现深度优先搜索遍历连通分量，用一个大小和 grid 相同的矩阵 visited 来记录当前节点是否被访问过
//      DFS过程中记录边界点存入数组 borders 中。
void dfs(vector<vector<int>>& grid, int x, int y, vector<vector<bool>>& visited, vector<pair<int, int>>& borders, int originalColor) {
    int m = grid.size(), n = grid[0].size();
    bool isBorder = false;
    int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    for (int i = 0; i < 4; i++) {
        int nx = dirs[i][0] + x, ny = dirs[i][1] + y;
        if (!(nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == originalColor)) {
            isBorder = true;
        } else if (!visited[nx][ny]) {
            visited[nx][ny] = true;
            dfs(grid, nx, ny, visited, borders, originalColor);
        }
    }
    if (isBorder) {
        borders.emplace_back(x, y);
    }
}
vector<vector<int>> colorBorderDFS(vector<vector<int>>& grid, int row, int col, int color) {
    int m = grid.size(), n = grid[0].size();
    vector<vector<bool>> visited(m, vector<bool>(n, false));
    vector<pair<int, int>> borders;

    int originalColor = grid[row][col];
    visited[row][col] = true;

    dfs(grid, row, col, visited, borders, originalColor);
    for (auto & point : borders) {
        grid[point.first][point.second] = color;
    }
    return grid;
}


// 6. 子岛屿的数量
//  这题的理解难度主要在：当岛屿 B 中所有陆地在岛屿 A 中也是陆地的时候，岛屿B是岛屿A的子岛屿。
//      反过来想：如果岛屿 B 存在一片陆地在岛屿 A 中是海水，B 就不是 A 的子岛屿
// 思路： 那么，只要遍历 grid2 中的所有岛屿，排除掉那些不可能是子岛的岛屿，剩下的就是子岛
// （注：这个思路和 计算封闭岛屿数量 类似，它是先排除靠边的岛屿，这次是排除不可能是子岛的岛屿 ）
int countSubIslands(vector<vector<char>>& grid1, vector<vector<char>>& grid2){
    int m = grid1.size(), n = grid1[0].size(); // 输入保证grid1和grid2一样大
    // 排除掉 grid2 中那些不可能是子岛的岛屿
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(grid1[i][j] == '0' && grid2[i][j] == '1'){
                // 这个岛屿肯定不是子岛，淹掉
                infect(grid2, i, j);
            }
        }
    }
    // 现在 grid2 中剩下的岛屿都是子岛，计算岛屿数量
    int subIsland = 0;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(grid2[i][j] == '1'){
                subIsland++;
                infect(grid2, i, j); // 从i，j开始把陆地变成海水
            }
        }
    }
    return subIsland;
}


// 7. 不同的岛屿数量
// 思路：显然，要统计不同的数量，就必须将二维数组进行转化，变成像字符串类型，然后利用 HashSet 去重这样来判断
//  但关键就是，如何将 二维数组转化成字符串，（联想到，这很类似于二叉树的序列化于反序列化，它就是将 二叉树和字符串互相转化）
// 所以，这里 只要保证从同一起点出发，dfs的遍历顺序是一样的就行。
//     比如：用 1，2，3，4 表示上下左右，-1，-2，-3，-4 表示上下左右的撤销
//        那么: "下，右，上"  ==>  "2，4，1" 这不就是序列化的结果
void serializeDFS(vector<vector<char>>& m, int i, int j, string& track, int dir){
    int row = m.size(), col = m[0].size();
    if(i <0 || i>= row || j < 0 || j >= col || m[i][j] == '0'){
        return ;
    }
    m[i][j] = '0';
    track.append(to_string(dir));
    track.push_back(',');

    serializeDFS(m, i-1, j, track, 1); // 上
    serializeDFS(m, i+1, j, track, 2); // 下
    serializeDFS(m, i, j-1, track, 3); // 左
    serializeDFS(m, i, j+1, track, 4); // 右

    track.append(to_string(-dir));
    track.push_back(',');
}
int numDistinctIslands(vector<vector<char>>& grid){
    int m = grid.size(), n = grid[0].size(); // 输入保证grid1和grid2一样大
    unordered_set<string> distinctIslands;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(grid[i][j] == '1'){
                string str;
                serializeDFS(grid, i, j, str, 0); // 从i，j开始把陆地变成海水, 初始方向填任意值都不影响正确性
                distinctIslands.insert(str);
            }
        }
    }
    return distinctIslands.size();
}




#endif //DATASTRUCT_ALGORITHM_DP_2D_M_H
/*
 * // 1.2 不同的路径III (hard)
有四种类型方格：
1 表示起始方格。且只有一个起始方格。
2 表示结束方格，且只有一个结束方格。
0 表示我们可以走过的空方格。
-1 表示我们无法跨越的障碍。
每一个无障碍方格都要通过一次，但是一条路径中不能重复通过同一个方格。
 * 回溯 ：O(m * n * 2^{m*n})其中m,n 是给定二维网格行与列的大小
class Solution {
public:
    void DFS(int x, int y, int zero_count, int& path_count, vector<vector<int>>& grid)
    {
        //> 判断是否越界
        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size())
            return;
        //> 为障碍，则结束
        if (grid[x][y] == -1)
            return ;
        //> 不仅要走到结束方格，还要每一个无障碍方格走一边
        if (grid[x][y] == 2 && zero_count != 0 )
            return;

        if (grid[x][y] == 2 && zero_count == 0)
        {
            path_count++;
            return ;
        }

        int temp = grid[x][y];
        //> 标记走过
        grid[x][y] = -1;
        //> 开始回溯
        DFS(x-1, y, zero_count-1 , path_count,grid);
        DFS(x+1, y, zero_count-1 , path_count,grid);
        DFS(x, y-1, zero_count-1 , path_count,grid);
        DFS(x, y+1, zero_count-1 , path_count,grid);
        grid[x][y] = temp;
    }
    int uniquePathsIII(vector<vector<int>>& grid) {
        //> 找到入口
        int x , y;
        path_count = 0;
        zero_count = 0;
        for (int i = 0; i < grid.size(); ++i)
        {
            for (int j = 0; j < grid[0].size(); ++j)
            {
                if (grid[i][j] == 1)
                {
                    x = i;
                    y = j;
                    zero_count++;
                }
                if (grid[i][j] == 0)
                    zero_count++;
            }
        }
        //> 参数，起始坐标x，y，当前还需走过的空方格，路径条数，二维网格
        DFS(x, y, zero_count, path_count,grid);
        return path_count;
    }
private:

    int path_count;
    int zero_count;
};
*/

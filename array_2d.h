//
// Created by kai.chen on 2021/12/22.
//      1. 转圈打印二维数组 1.1 螺旋矩阵
//      2. 将二维数组顺时针旋转90度
//
//      3. 搜索二维矩阵
//          3.1 搜索二维矩阵II
//          3.2 搜索有序矩阵中第k小的元素
//      4. 稀疏矩阵的乘法
//      5. 矩阵置零
//
//      6. Z字形变换
//      7. 对角线遍历
#ifndef DATASTRUCT_ALGORITHM_ARRAY_2D_H
#define DATASTRUCT_ALGORITHM_ARRAY_2D_H
#include <vector>
using namespace std;

// 1.转圈打印二维数组
// 思路：从易于理解的角度，矩阵分圈处理，
//      从左上角x(x1, y1)到右下角y(x2,y2)为一个圈，每次只需要缩小圈直到结束就好了
//      1   2   3  4
//      5          8
//      9  10  11  12
void circlePrint(vector<vector<int>>& m, int x1, int y1, int x2, int y2){
    if(x1 == x2){ // 只有一行
        for(int y = y1; y <= y2; y++){
            cout<<m[x1][y]<<" ";
        }
    }
    else if(y1 == y2){ // 只有一列
        for(int x = x1; x <= x2; x++){
            cout<<m[x][y1]<<" ";
        }
    }
    else {
        int x = x1, y = y1;
        while(y < y2) { cout<<m[x][y]<<" "; y++;} // 1 2 3     ==> x = x1, y = y2
        while(x < x2) { cout<<m[x][y]<<" "; x++;} // 4 8       ==> x = x2, y = y2
        while(y > y1) { cout<<m[x][y]<<" "; y--;} // 12 11 10  ==> x = x2, y = y1
        while(x > x1) { cout<<m[x][y]<<" "; x--;} // 9 5       ==> x = x1, y = y1
    }
}

void sprialOrderPrint(vector<vector<int>>& m){
    if (m.size() == 0 || m[0].size() == 0) return ;
    int x1 = 0, y1 = 0;
    int x2 = m.size()-1, y2 = m[0].size()-1;
    while(x1 <= x2 && y1 <= y2){
        circlePrint(m, x1, y1, x2, y2);
        x1++,y1++;
        x2--,y2--;
    }
}



// 2. 将二维数组顺时针旋转90度
//      1   2   3   4
//      5           8
//      9           12
//      13  14  15  16
// 分圈处理，每一圈中：1，4，16，13为一组，需要旋转一维度
void circleRotate(vector<vector<int>>& m, int x1, int y1, int x2, int y2) {
    int d = y2 - y1; // 一圈中待旋转的组数量
    for(int i = 0; i < d; i++){
        int tmp = m[x1][y1+i];      // 1
        m[x1][y1+i] = m[x2-i][y1];  // 13->1
        m[x2-i][y1] = m[x2][y2-i];  // 16->13
        m[x2][y2-i] = m[x1+i][y2];  // 4->16
        m[x1+i][y2] = tmp;          // 1->4
    }
}
void rotateMatrix(vector<vector<int>>& m){
    int x1 = 0, y1 = 0;
    int x2 = m.size()-1, y2 = m[0].size()-1;
    while(x1 < x2){
        circleRotate(m, x1, y1, x2, y2);
        x1++,y1++;
        x2--,y2--;
    }
}

void testPrintOrder(){
    int matrix[][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    int x = sizeof(matrix)/sizeof(matrix[0]);
    int y = sizeof(matrix[0])/sizeof(int);
    vector<vector<int>> m(x, vector<int>(y));
    for(int i = 0; i < x; i++){
        for(int j = 0; j < y; j++){
            m[i][j] = matrix[i][j];
        }
    }

    for(int i = 0; i < x; i++){
        for(int j = 0; j < y; j++){
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }
    //sprialOrderPrint(m);
    rotateMatrix(m);
    for(int i = 0; i < x; i++){
        for(int j = 0; j < y; j++){
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }
}


// 3. 搜索二维矩阵
// 题目描述：编写一个高效的算法来判断m x n矩阵中，是否存在一个目标值。该矩阵具有如下特性：
//      每行升序排列
//      每行的第一个整数大于前一行的最后一个整数。
// 思路：
//      若将矩阵每一行拼接在上一行的末尾，则会得到一个升序数组，我们可以在该数组上二分找到目标元素。
//        // 行列的表示可以用 / 运算和 % 运算得到
//         时间复杂度O(log M*N)
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int m = matrix.size(), n = matrix[0].size();

    int l = 0, r = m * n - 1;
    while (l <= r) {
        int mid = (r - l) / 2 + l;
        int x = matrix[mid / n][mid % n]; // 行列的表示
        if(x == target){
            return true;
        }
        else if (x < target) {
            l = mid + 1;
        } else if (x > target) {
            r = mid - 1;
        }
    }
    return false;
} // 时间复杂度: O(log(m*n))
// 空间: O(1)

// 3.1 搜索二维矩阵II
// 题目描述：编写一个高效的算法来判断m x n矩阵中，是否存在一个目标值。该矩阵具有如下特性：
//      每行升序排列，每列升序排列。
//      但不保证 每行的第一个整数大于前一行的最后一个整数。
// 思路：
//     遍历行，然后再对列进行二分。 时间复杂度O(m * logN)
bool searchMatrixII(vector<vector<int>>& matrix, int target) {
    int m = matrix.size(), n = matrix[0].size();
    for (int i = 0; i < m; i++) {
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (r-l)/2 + l;
            int x = matrix[i][mid];
            if(x == target){
                return true;
            }else if (x < target) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
    }

    return false;
}

// 思路：Z形查找 时间复杂度O(m+n)
// 从右上角 (0, n-1)开始搜。直到搜到左下角
//  x行坐标坐标每次加++，y坐标每次--
bool searchMatrixZ(vector<vector<int>>& matrix, int target) {
    int m = matrix.size(), n = matrix[0].size();
    int row = 0, col = n - 1;
    // 从右上角(0,n-1)开始扫描
    while (row < m && col >= 0) { // 直到搜到左下角(m,0)
        int x = matrix[row][col];
        if (x == target) {
            return true;
        }
        if (x > target) { // target 小于当前扫描值
            col--; // 列值减一，让值更小
        }else {
            row++; // 行值加一，让值更大
        }
    }
    return false;
}

// 3.2 搜索有序矩阵中第k小的元素
// 题目描述：编写一个高效的算法来找出矩阵中第 k 小元素。该矩阵具有如下特性：
//      每行升序排列，每列升序排列。
//      但不保证 每行的第一个整数大于前一行的最后一个整数。
// 思路：暴力解法是对n^2个数排序，时间复杂度是n^2 *logn
//  二分法：时间复杂度：O(nlog(r−l))，二分查找进行次数为 O(log(r−l))， r是数组最大值，l是数组最小值
bool check(vector<vector<int>>& matrix, int mid, int k, int n) {
    // 检查第k小的元素是否在左区间  将matrix化为一维数组后的小于mid的元素个数 是否大于 k .是的话在左区间
    // 要从 mid 分，从左下角元素开始，找到第一个
    int i = n - 1, j = 0; // 从第 j 列的第 i个元素开始搜，j：列号
    int cnt = 0;
    while (i >= 0 && j < n) {
        if (matrix[i][j] <= mid) {
            cnt += i + 1; // j 这一整列的元素都小于mid
            if( cnt >= k) return true; // 剪枝 发现cnt个数大于k提前退出
            j++; // 向右
        } else {
            i--; // 向上
        }
    }
    return cnt >= k;
}
int kthSmallest(vector<vector<int>>& matrix, int k) {
    int n = matrix.size();
    if (k <= 0 || k > n*n) return -1;
    int left = matrix[0][0];
    int right = matrix[n - 1][n - 1];
    while (left < right) {
        int mid = left + ((right - left) >> 1);
        if (check(matrix, mid, k, n)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// 4. 稀疏矩阵的乘法
// 题目描述：两个 稀疏矩阵 A 和 B，请你返回 AB 的结果。可以默认 A 的列数等于 B 的行数。
// C = AB = {a(1,1)b(1,1)+a(1,2)b(2,1)+a(1,3)b(3,1), a(1,1)b(1,2)+a(1,2)b(2,2)+a(1,3)b(3,2) ...
//输入：
//A = [
//  [ 1, 0, 0],
//  [-1, 0, 3]
//]
//B = [
//  [ 7, 0, 0 ],
//  [ 0, 0, 0 ],
//  [ 0, 0, 1 ]
//]
//输出：
//     |  1 0 0 |   | 7 0 0 |   |  7 0 0 |
//AB = | -1 0 3 | x | 0 0 0 | = | -7 0 3 |
//                  | 0 0 1 |
// 思路：
//    暴力解法是：3层for循环直接相乘
//    较好的做法是：只选取不为0的行列相乘，遍历判断A的每一行，B的每一列
vector<vector<int>> multiply(vector<vector<int>>& A, vector<vector<int>>& B) {
    vector<vector<int>> res(A.size(), vector<int>(B[0].size(), 0));
    int i, j, k, sum;
    for(i = 0; i < A.size(); i++){ // A的每一行
        for(j = 0; j < B[0].size(); j++){ // B的每一列
            sum = 0;
            for(k = 0; k < A[0].size(); k++){
                sum += A[i][k]*B[k][j];
            }
            res[i][j] = sum;
        }
    }
    return res;
}
vector<vector<int>> multiply2(vector<vector<int>>& A, vector<vector<int>>& B) {
    vector<bool> rowIsAll0(A.size(),true);
    vector<bool> colIsAll0(B[0].size(),true);

    int i, j, k, sum;
    bool flag = false;
    // 遍历判断A的每一行
    for(i = 0; i < A.size(); ++i){
        flag = false;
        for(j = 0; j < A[0].size(); ++j){
            if(A[i][j]){
                flag = true;
                break;
            }
        }
        if(flag) rowIsAll0[i] = false;
    }
    // 遍历判断B的每一列
    for(j = 0; j < B[0].size(); ++j){
        flag = false;
        for(i = 0; i < B.size(); ++i){
            if(B[i][j]){
                flag = true;
                break;
            }
        }
        if(flag) colIsAll0[j] = false;
    }

    vector<vector<int>> ans(A.size(), vector<int>(B[0].size(), 0));
    for(i = 0; i < A.size(); ++i){
        for(j = 0; j < B[0].size(); ++j){
            // 判断该行列，如果都为0可以直接跳过不用算。
            if(rowIsAll0[i] || colIsAll0[j]) continue;
            sum = 0;
            for(k = 0; k < A[0].size(); ++k){
                sum += A[i][k]*B[k][j];
            }
            ans[i][j] = sum;
        }
    }
    return ans;
}


// 5. 矩阵置零
// 题目描述：给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
// 思路：
//   （1）直接使用一个行标记数组，一个列标记数组。分别标记每行或每列是否有0，然后根据标记数组更新矩阵。空间O(m+n)
//    (2) 借用矩阵的第一行和第一列代替方法一中的两个标记数组
//       这样 不知道 第一行/第一列 是否原本包含 0。
//          可以使用两个标记变量来 先分别记录第1行是否为零、第1列是否为零
//
void setZeroes1(vector<vector<int>>& matrix) {
    int m = matrix.size();
    int n = matrix[0].size();
    vector<int> row(m), col(n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (!matrix[i][j]) row[i] = col[j] = true;
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (row[i] || col[j]) matrix[i][j] = 0;
        }
    }
} // 时间 O(m*n), 空间O(m+n)

void setZeroes2(vector<vector<int>>& matrix) {
    int m = matrix.size();
    int n = matrix[0].size();
    // 两个标记变量 先记录第1行是否为零、第1列是否为零
    int flag_col0 = false, flag_row0 = false;
    for (int i = 0; i < m; i++) {
        if (!matrix[i][0]) {
            flag_col0 = true;
        }
    }
    for (int j = 0; j < n; j++) {
        if (!matrix[0][j]) {
            flag_row0 = true;
        }
    }

    // 借用矩阵的第一行和第一列代替方法一中的两个标记数组
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (!matrix[i][j]) {
                matrix[i][0] = matrix[0][j] = 0;
            }
        }
    }
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (!matrix[i][0] || !matrix[0][j]) {
                matrix[i][j] = 0;
            }
        }
    }

    // 如果第一列原本就包含0，需要置零
    if (flag_col0) {
        for (int i = 0; i < m; i++) {
            matrix[i][0] = 0;
        }
    }
    // 如果第一行原本就包含0，需要置零
    if (flag_row0) {
        for (int j = 0; j < n; j++) {
            matrix[0][j] = 0;
        }
    }
}

// 6. Z字形变换
// 题目描述：输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下；输出 PAHNAPLSIIGYIR
// 思路：直接模拟
string convert(string s, int numRows) {
    if (numRows == 1) return s;

    int step = numRows * 2 - 2; // 间距
    int index = 0;  // 记录s的下标
    int len = s.length();
    int add = 0; // 真实的间距
    string ret;

    for (int i = 0; i < numRows; i++){ // i表示行号
        index = i;
        add = i * 2;
        while (index < len){//超出字符串长度计算下一层
            ret += s[index]; // 当前行的第一个字母
            add = step - add;// 第一次间距是step -2*i，第二次是2*i,
            index += (i == 0 || i == numRows-1) ? step : add; // 0行和最后一行使用step间距，其余使用add间距
        }
    }
    return ret;
} // 时间 O(n), 空间O(1)

// 7. 对角线遍历
vector<int> finddiagonalOrder(vector<vector<int>>& matrix){
    vector<int> nums;
    int m = matrix.size();
    int n = matrix[0].size();

    for (int i = 0; i < m + n; ){ // i 是 x + y 的和
        // 第 1 3 5 ... 趟
        int x1 = (i < m) ? i : m - 1;	// 确定 x y 的初始值
        int y1 = i - x1;
        while (x1 >= 0 && y1 <= n-1){
            nums.push_back(matrix[x1][y1]);
            x1--; // 移动方向为左下 到 右上
            y1++;
        }
        i++;

        if (i >= m + n) break;
        // 第 2 4 6 ... 趟
        int y2 = (i < n) ? i : n - 1;	// 确定 x y 的初始值
        int x2 = i - y2;
        while (y2 >= 0 && x2 <= m-1){
            nums.push_back(matrix[x2][y2]);
            x2++; // 移动方向为右上 到 左下
            y2--;
        }
        i++;
    }
    return nums;
}




#endif //DATASTRUCT_ALGORITHM_ARRAY_2D_H

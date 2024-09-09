//
// Created by kai.chen on 2021/12/21.
// 二维回溯 显式二维矩阵题
//      1. N皇后
//      2. 单词搜索
//
//      . 解数独 见 bfs_dfs.h d5

#ifndef DATASTRUCT_ALGORITHM_BACKTRAK_2DM_H
#define DATASTRUCT_ALGORITHM_BACKTRAK_2DM_H
#include <string>
#include <vector>
using namespace std;

// 1. N皇后
vector<vector<string>> res;
bool isValid(const vector<string> &board, int row,  int col){
    int n = board.size();
    for(int i = 0; i < row; i++){
        if(board[i][col] == 'Q') return false;
    }
    for(int i = row-1, j = col-1; i >= 0 && j >= 0; i--,j--) {
        if(board[i][j] == 'Q') return false;
    }
    for(int i = row-1, j = col+1; i >= 0 && j < n; i--,j++) {
        if(board[i][j] == 'Q') return false;
    }
    return true;
}
void backtrack(vector<string> &board, int row){
    if(row == board.size()){
        res.push_back(board);
        return;
    }
    int n = board.size();
    for(int i =0; i< n; i++){
        if(!isValid(board, row, i))
            continue;
        board[row][i] = 'Q';
        backtrack(board, row+1);
        board[row][i] = '.';
    }
}
vector<vector<string>> solveNQueens(int n) {
    vector<string> board(n, string(n,'.'));
    backtrack(board, 0);
    return res;
}

// 2. 单词搜索
// 题目描述： m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
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




#endif //DATASTRUCT_ALGORITHM_BACKTRAK_2DM_H

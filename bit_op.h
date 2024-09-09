//
// Created by kai.chen on 2021/12/31.
//  位运算
//      1. 统计二进制中 1 的个数
//      2. 不用额外变量"交换"两个数
//      3. 不做任何"比较"找出更大的数
//       3.1 不用加减乘除做加法
//       3.2 转换为k进制数
//
//      4. 其他数都出现偶数次，找出"出现奇数次"的数(只有一个)
//          变体4.1：有两个出现奇数次，其他出现偶数次，找出这两个数
//      5. 找出只出现一次的数，其他都出现k次
//
//      6. 一和零 (请见 dp_hd.h 01背包问题)
//
//      7. 字符串相加、36进制加法 (请键two_sum.h)
//
#ifndef DATASTRUCT_ALGORITHM_BIT_OP_H
#define DATASTRUCT_ALGORITHM_BIT_OP_H

// 1. 统计二进制中 1 的个数
// 思路：
//    (1) 常规思路:  每次 与1相"与"，向右移一位。(这样每次判断最后一位是否为1，并且每次抹掉一个1如果是的话，时间复杂度是 整数二进制长度32)
//    (2) 一般思路:  每次 与x-1相"与"，相当于每次去掉了最右侧一个1。(时间复杂度是 二进制中1的个数)
//    (3) 思路3:  每次 x - ( x & (~x+1)), 也相当于每次去掉了最右侧一个1    注：这题还有很多逆天算法：如平行算法，MIT hackmem算法...

int count1(int x){
    int count = 0;
    //  for (int i = 0; i < 32; i++) {
    //        if (((x >> i) & 1) == 1) {
    //            count++;
    //        }
    //    }
    while(x != 0){
        count = count + x&1;
        x = x>>1;
    }
    return count;
}
// 推荐写法：
int count2(int x){
    int count = 0;
    while(x != 0){
        count++;
        x = x&(x-1);
    }
    return count;
}
// __builtin_popcount: 查表实现，时间复杂度为 O (lg N) , N 为位数
int count2_2(int x){
    return __builtin_popcount(x);
}
// int bitCount(int i) {
//    //table是0到15转化为二进制时1的个数
//    int table[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
//    int count = 0;
//    while (i != 0) {//通过每4位计算一次，求出包含1的个数
//        count += table[i & 0xf];
//        i >>>= 4;
//    }
//    return count;
//}


// follow up: 计算前n个二进制数中1的个数
// 朴素做法
vector<int> countBits1(int n) {
    vector<int> res(n+1);
    for(int i = 0; i < n+1; i++){
        res[i] = __builtin_popcount(i);
    }
    return res;
}



// 2. 不用额外变量交换两个数
//  思路： "异或"运算
void swap_yihuo(int* a, int *b){
    *a = *a ^ *b;
    *b = *a ^ *b;
    *a = *a ^ *b;
}
void testswap(){
    int a = 1, b =2;
    swap_yihuo(&a, &b);
    cout<<"a = "<<a <<endl<<"b = "<<b<<endl;
}

// 3. 不做任何比较找出更大的数
// 思路：判断 a-b 的符号
int sign(int x){ // 返回x的符号，正数返回1，负数返回0
    return ((x>>31) & 1) ^ 1; // 找出符号位，和1异或表示取反
}
int getMax(int a, int b){
    int c = a-b;
    int sgA = sign(c); // sgA如果a 更大才为1，否则为0
    int sgB = sgA ^ 1; // 和1异或表示取反
    return sgA * a + sgB * b;
}

// 3.1 不用加减乘除做加法
// 思路："无进位和" 与 异或运算 规律相同，"进位" 和 与运算 规律相同（并需左移一位
// a(i)	b(i) 无进位和n(i) 进位c(i+1)
// 0	 0	 0	 0
// 0	 1	 1	 0
// 1	 0	 1	 0
// 1	 1	 0	 1

// 时间复杂度 O(1)： 最差情况下需循环 32 次
int add(int a, int b) {
    // s = a+b ⇒s = n+c (其中n=a^b，c=(a&b)<<1)
    while(b != 0) { // 当进位为 0 时不需要继续加了
        int c = (a & b) << 1;  // c = 进位
        a ^= b; // a = 非进位和
        b = c; // b = 进位
    }

    return a;
}

// 3.2 转换为k进制数
vector<int> convertTokjzNum(int n, int k){
    vector<int> bitNumkjz(32);
    int idx = 0;
    while(n != 0){
        bitNumkjz[idx++] = n % k;
        n = n/k;
    }
    return bitNumkjz;
}
int convertFromkjzNum(vector<int>& nums, int k){
    int res = 0;
    for(int i = nums.size()-1; i >= 0; i--){
        res = res*k + nums[i];
    }
    return res;
}

// 4. 其他数都出现偶数次，找出出现奇数次的数
// 思路：异或运算的"交换律"和"结合律", C B D A A B C => A A B B C C D => D
void printOddTimesNum1(vector<int>& arr){
    int res = 0;
    for(int cur : arr){
        res = res ^ cur;
    }
    cout << res << endl;
}

// 4.1 如果有两个数出现奇数次
//
// 思路：    直观的解法是哈希表记录出现次数，遍历一遍哈希表就可以找出来，时间O(N),空间O(N)
//      用异或最后的结果就是 A^B（假设剩下的两个数是A和B）
//      用题1 的思路3：找出A_B右侧的第一个等于1的位; 在这一位上一定是 A等于0， B等于1。 (基于异或的特性)
void printOddTimesNum2(vector<int>& arr){
    int A_B = 0, A = 0, B = 0;
    for(int cur : arr){
        A_B = A_B^cur;
    }
    // 到这 A_B = A ^ B
    //  找出A_B右侧的第一个等于1的位; 在这一位上一定是 A等于0， B等于1。 (基于异或的特性)
    int rightOneBit;
    if(A_B == INT_MIN) rightOneBit = A_B; // -2147483648是补码表示，只有一个符号位为1
    else rightOneBit = A_B & (~A_B + 1); //（A & (A取反+1) 的结果就是最后一个1的十进制表示,注意防溢出 ）

    // 再遍历一次数组，每次只和第k位上（第一个等于1的位）是1的整数异或，其他的数忽略。就能得到A
    for(int i: arr){
        if((i & rightOneBit) != 0){
            A = A^i;
        }
    }
    B = A_B ^ A;
    cout<<"A = "<<A <<endl<<"B = "<<B<<endl;
}

// 5. 找出只出现一次的数，其他都出现k次
// 思路：
//   关键原理： k个相同的k进制数无进位相加，结果的每一位上都将是 k的倍数
//   具体做法：遍历arr，每一个数都转成k进制数，然后进行无进位相加。结束的最后结果 从k进制转成十进制 就是出现一次的数的结果
//      第二种更直观的做法：直接判断整型数字每个二进制位
int OnceNum(vector<int>& nums, int k){
    vector<int> plusRes(32);
    for(int i = 0; i < nums.size(); i++){
        // 遍历，进行无进位相加
        vector<int> currKjzNum = convertTokjzNum(nums[i], k);
        for(int j = 0; j < 32; j++){
            plusRes[i] = (plusRes[i] + currKjzNum[i]) % k;
        }
    }
    return convertFromkjzNum(plusRes, k);
}
int OnceNum2(vector<int>& nums, int k) {
    int ans = 0;
    for(int i = 0; i < 32; i++){
        int cnt = 0;
        for(const auto& num : nums){
            cnt += (num >> i) & 1;
        }
        if(cnt % k) ans |= (1 << i);
    }
    return ans;
}


// 6. 一和零 请见 dp_hd.h 01背包问题
// 题目描述L：给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
//     请找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。
//      输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
//      输出：4
//      解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。


#endif //DATASTRUCT_ALGORITHM_BIT_OP_H

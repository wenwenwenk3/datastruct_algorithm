//
// Created by kai.chen on 2021/12/12.
//
// 1. 最小覆盖子串
// 2. 判断包含字符串的排列
// 3. 找所有字母异位词
//    3.1 交错字符串 见dp_hd.h
// 4. 最长无重复子串  延伸: 最多包含两个不同字符的最长子串
//     4.1 重复的DNA序列
//     4.2 最长重复子串
//     4.3 区间是否存在重复数据
//
// 5. 删掉一个元素以后全为1的最长子数组(LC1493)
// 6. 最大连续1的个数 (LC1004) （最多将 K 个值从 0 变成 1）
//   0和1的个数相同的子数组 array_presum
//
// 7. 替换后的最长重复字符串长度(LC424)
// 8. 尽可能使字符串相等LC
// 9. 统计优美数组
//
// 10. 长度最小的 和>target的子数组
// 11. 和为target的连续正整数序列
// 12. 乘积小于k的子数组 [经典题]
// 13. 得到k个黑块的最少涂色次数
//
//    滑动窗口最大值 （解法见 monqueue单调队列） 滑动窗口的中位数 find_kth_largest.h
#ifndef DATASTRUCT_ALGORITHM_SLIDING_WINDOW_H
#define DATASTRUCT_ALGORITHM_SLIDING_WINDOW_H
#include <unordered_map>
#include <string>
using namespace std;

// 1. 最小覆盖子串
// 问题描述：两个字符串 S 和 T ，请在S中找出包含T中全部字母的最短子串
// 暴力解法是 两个for循环遍历所有可能区间，检测区间是否包含所有字母，是的话检查是否需要更新答案。复杂度N3
// 滑动窗口经典做法：双指针[left, right)，先不断的增加right直到包含所有字符
//      此时停止增加right，开始增加left缩短窗口，直到不符合要求。每次增加left更新结果
//      重复这两个步骤，直到right达到s的尾部
string min_window(string s, string t){
    // 需要凑齐的字符，窗口中的字符
    unordered_map<char, int> need, window;
    for(auto c : t){
        need[c]++;
    }

    int left = 0, right = 0;
    int valid = 0;
    int start = 0, minlen = INT_MAX; // 记录最小子串的位置和长度
    while(right < s.length()){
        char c = s[right]; // 每次移入字符c
        right++; // 窗口右移

        if(need.count(c)){ // c是need的字符，更新窗口数据
            window[c]++;
            if(window[c] == need[c]){
                valid++; //表示window中有一个字符的个数和need匹配了
            }
        }
        // 当满足条件了，收缩左侧
        while(valid == need.size()) {
            if (right - left < minlen) {
                start = left;
                minlen = right - left;
            }
            char d = s[left]; // 待移出字符d
            left++; // 窗口左侧向后移动

            if (need.count(d)) { // 移出的字符是need的
                if (window[d] == need[d]) {
                    valid--; // 意味着window中字符d的个数少了
                }
                window[d]--;
            }
        }
    }
    return minlen == INT_MAX ? "": s.substr(start, minlen);
}

void test_minwindow(){
    string s = "abcde";
    string t = "bcd";

    string subst = min_window(s, t);
    cout<<subst<<endl;
}


// 2. 判断包含字符串的排列
// 问题描述：两个字符串S，T， 判断S是否包含T的排列，也就是判断S中是否存在子串是T的一个全排列
// 典型滑动窗口解法：双指针[left, right)，不断移动right直到right-left足够长。
//      一旦发现符合条件valid==need.size()，说明存在一个合法排列，return true.
bool checkInclusion(string s, string t){
    unordered_map<char, int> need, window;
    for(char c:t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;
    while(right < s.length()){
        char c = s[right];
        right++;

        if(need.count(c)){ // c是need的字符，更新窗口数据
            window[c]++;
            if(window[c] == need[c]){
                valid++; //表示window中有一个字符的个数和need匹配了
            }
        }

        while(right - left >= t.length()){
            if(valid == need.size()){
                return true; // 找到满足条件
            }
            char d = s[left];
            left++;
            if(need.count(d)){
                if(window[d] == need[d]){
                    valid--;
                }
                window[d]--;
            }
        }
    }
    return false;
}

// 3. 找所有字母异位词
// 题目描述：两个字符串S，T，找到S中所有是T的字母异位词的子串，返回这些子串的起始索引。
// （若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。其实就是全排列,
//        这里有开始索引自然就可以找到子串，因为异位词的长度是固定的）
// 滑动窗口解法：
//      不断移动right直到right-left足够长。
////      一旦发现符合条件valid==need.size()，说明存在一个合法异位字母词，记录一下.
vector<int> findAnagrams(string s, string t){
    unordered_map<char, int> need, window;
    for(char c:t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;
    vector<int> res;
    while(right < s.length()){
        char c = s[right];
        right++;

        if(need.count(c)){ // 如果移入的字母c是need需要的，更新window
            window[c]++;
            if(window[c] == need[c]){
                valid++; //表示window中有一个字符的个数和need匹配了
            }
        }

        // 判断左侧窗口是否需要收缩
        while(right - left >= t.length()){
            if(valid == need.size()){
                res.push_back(left); // 找到满足条件
            }
            char d = s[left];
            left++;
            if(need.count(d)){// 如果移出的字母d是need需要的，更新window
                if(window[d] == need[d]){
                    valid--;
                }
                window[d]--;
            }
        }
    }
    return res;
}

// 3.1. 拓展: 判断 t 是否是 s 的字母异位词
bool isAnagram(string s, string t) {
    if (s.length() != t.length()) {
        return false;
    }
    vector<int> table(26, 0);
    for (auto& ch: s) {
        table[ch - 'a']++;
    }
    for (auto& ch: t) {
        table[ch - 'a']--;
        if (table[ch - 'a'] < 0) {
            // 用一个hash表保存每个字母个数，当t有的字符，s没有 或 t中某个字母的个数多于s， 直接false
            return false;
        }
    }
    return true;
}


// 4. 最长无重复子串
// 题目描述：输入一个字符串s，计算s中不包含重复字符的最长子串长度
//      不断移动right直到s的尾部
//      左侧需要收缩的条件是新加入窗口的字符c是重复数字
int lengthOfLongestSubstring(string s){
    unordered_map<char, int> window;

    int left = 0, right = 0;
    int maxlen = 0;
    while(right < s.length()){
        char c = s[right];
        right++;

        window[c]++; // 更新窗口数据

        // 判断左侧窗口是否需要收缩
        while(window[c] > 1){ // 左侧需要收缩的条件是新加入窗口的字符c是已经包含重复字母
            char d = s[left];
            left++;

            window[d]--; // 更新
        }

        maxlen = max(maxlen, right-left);
    }
    return maxlen;
} // 时间复杂度: O(N)
// 空间复杂度: O(Σ) 字符集的大小，默认为所有 ASCII 码在 [0, 128)[0,128) 内的字符

void test_lenLSubstring(){
    string s = "abcdbcdbcde";
    cout<<lengthOfLongestSubstring(s)<< endl;
}

// 延伸 求最多包含两个不同字符的最长子串： 窗口收缩的条件就是size>2
//      判断 map 里面的元素是不是大于2个，如果是，说明得删除一个，找出三个元素 value 最小的那个进行删除；
int lengthOfLongestSubstringII(string s){
    unordered_map<char, int> window;

    int left = 0, right = 0;
    int maxlen = 0;
    while(right < s.length()){
        char c = s[right];
        right++;

        window[c]++; // 更新窗口数据

        // 判断左侧窗口是否需要收缩
        while(window.size() > 2){ // 左侧需要收缩的条件是新加入窗口的字符c是已经包含重复字母
            char d = s[left];
            left++;

            window[d]--; // 更新
            if(window[d] == 0) window.erase(d);
        }

        maxlen = max(maxlen, right-left);
    }
    return maxlen;
} // 时间复杂度: O(N)

// 4.0 至少有 K 个重复字符的最长子串
// 这里滑动窗口挺难想的，需要枚举字符出现次数
int longestSubstring(string s, int k) {
    int ret = 0;
    int n = s.length();
    for (int t = 1; t <= 26; t++) {
        int l = 0, r = 0;
        vector<int> cnt(26, 0);
        int tot = 0;
        int less = 0;
        while (r < n) {
            cnt[s[r] - 'a']++;
            if (cnt[s[r] - 'a'] == 1) {
                tot++;
                less++;
            }
            if (cnt[s[r] - 'a'] == k) {
                less--;
            }

            while (tot > t) {
                cnt[s[l] - 'a']--;
                if (cnt[s[l] - 'a'] == k - 1) {
                    less++;
                }
                if (cnt[s[l] - 'a'] == 0) {
                    tot--;
                    less--;
                }
                l++;
            }
            if (less == 0) {
                ret = max(ret, r - l + 1);
            }
            r++;
        }
    }
    return ret;
} // 时间复杂度：O(N⋅∣Σ∣+∣Σ∣) 其中 N 为字符串的长度，Σ 为字符集数量=26


// 4.1 重复的DNA序列
// DNA序列由字母ACGT组成，求返回所有在 DNA 分子中出现不止一次的 长度为 10 的序列(子字符串)
vector<string> findRepeatedDnaSequences(const string& s) {
    unordered_map<string, int> freq;
    string temp;
    vector<string> res;
    int n = s.length();
    for(int i = 0; i <= (n-10); i++){
        temp = s.substr(i, 10);
        freq[temp]++;
        if(freq[temp] == 2){
            res.push_back(temp);
        }
    }
    return res;
}
void test_findRepeatedDnaSequences(){
    string s = "A";
    vector<string> res = findRepeatedDnaSequences(s);
    for(const auto&str : res){
        cout<< str<<endl;
    }
}

// 4.3 区间k内是否存在重复数据
// 题目描述：判断数组中是否存在两个 不同的索引 i 和 j ，满足 nums[i] == nums[j] 且 abs(i - j) <= k
bool containsNearbyDuplicate1(vector<int>& nums, int k) {
    unordered_set<int> window;
    int l = 0, r = 0;
    while(r < nums.size()){
        if (window.count(nums[r])) {
            return true;
        }
        window.insert(nums[r]);
        r++;
        if(r -l > k) {
            window.erase(nums[l]);
            l++;
        }
    }
    return false;
}
// 稍微优雅一点的写法
bool containsNearbyDuplicate(vector<int>& nums, int k) {
    int n = nums.size();
    unordered_set<int> set;
    for(int i = 0; i < n; ++i){
        if(i > k) set.erase(nums[i - k - 1]);
        if(set.count(nums[i])) return true;
        set.insert(nums[i]);
    }
    return false;
}

// 4.4 区间k内是否存在值之差小于t的数据
// 题目描述：请你判断是否存在 两个不同下标 i 和 j，使得 abs(nums[i] - nums[j]) <= t ，同时又满足 abs(i - j) <= k 。
// 思路：滑动窗口
//   为了维护快速查找是否有值落在[x-t,x+t]范围内，可以使用一个有序集合作为窗口
//  在有序集合中查找大于等于x−t 的最小的元素 y，如果 y 存在，且 y≤x+t，我们就找到了一对符合条件的元素。
bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
    set<long> rec;
    for (int i = 0; i < nums.size(); i++) {
        auto iter = rec.lower_bound((long)nums[i]-t); // 防止int溢出
        if (iter != rec.end() && *iter <= (long)nums[i]+t) {
            return true;
        }
        rec.insert(nums[i]);
        if (i >= k) {
            rec.erase(nums[i - k]);
        }
    }
    return false;
}   // 时间复杂度：O(n * log(k)): 当然, k<n时是有序集合元素个数不到n, 二分复杂度是logn 每个元素至多被插入有序集合和从有序集合中删除一次，每次二分查找操作时间复杂度均为O(log(min(n,k))。
// 空间复杂度：O(min(n,k))


// 5. 删掉一个元素以后全为 1 的最长子数组
// 输入：nums = [1,1,0,1], 输出：3, 解释：删掉位置 2 的数后，[1,1,1] 包含 3 个 1 。
// 思路：滑动窗口, 求最多有一个0的最大窗口
int longestSubarray(vector<int>& nums) {
    int left=0,right=0;
    int n=nums.size(),res=0,count=0; // count 用来统计窗口内 0 的个数
    while(right<n){
        count+=nums[right]==0;
        // 当窗口内 0 的个数超过一个时，我们需要缩小窗口
        while(count>1){
            if(nums[left]==0){
                count--;
            }
            left++;
        }
        // 当窗口内 0 的个数小于等于一个时，记录最大窗口长度
        res=max(res, right-left+1);
        right++;
    }
    // 由于我们算的是，删掉一个0后的最长数组长度，所以最后结果还要减 1
    return res-1;
}


// 6. 最大连续1的个数III（将 K 个值从 0 变成 1）
// 题目描述：给定一个由若干 0 和 1 组成的数组 A，我们最多可以将 K 个值从 0 变成 1 。
//      返回仅包含 1 的最长（连续）子数组的长度。
//      如 输入：A = [1,1,1,0,0,0,1,1,1,1,0], K = 2  输出：6
// 思路：滑动窗口，求最多有K个0的最大窗口
int longestOnes(vector<int>& A, int K) {
    int left=0,right=0;
    int res=0,n=A.size(), count=0; //count用来统计窗口中0的个数

    while(right<n){
        count+=A[right]==0;
        while(count > K)//当窗口内0的个数大于K时，需要缩小窗口
        {
            if(A[left]==0){
                count--;
            }
            left++;
        }
        // 当窗口内 0 的个数小于等于K个时，记录最大窗口长度
        res=max(res, right-left+1);
        right++;
    }
    return res;
}
// 6.1 求最大连续1的个数 （最简单版本）
int findMaxConsecutiveOnes(vector<int>& nums) {
    int maxCount = 0, count = 0;
    int n = nums.size();
    for (int i = 0; i < n; i++) {
        if (nums[i] == 1) {
            count++;
        } else {
            maxCount = max(maxCount, count);
            count = 0;
        }
    }
    maxCount = max(maxCount, count);
    return maxCount;
}

// 7. 替换后的最长重复字符串长度(LC424)
// 题目描述：给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，
//      总共可最多替换 k 次。在执行上述操作后，找到包含重复字母的最长子串的长度
// 思路：滑动窗口
//   关键理解： 需要替换的字符个数就是当前窗口的大小减去窗口中数量最多的字符的数量
int characterReplacement(string s, int k) {
    int count[26]={0};//建立字符->字符数量的映射

    int left=0,right=0;
    int res=0,maxCount=0; //maxCount当前窗口内的最多字符的个数

    while(right<s.size())
    {
        count[s[right]-'A']++;
        maxCount = max(maxCount, count[s[right]-'A']); //当前窗口内的最多字符的个数
        // 需要替换的字符个数 > k时，怎么也不满足条件, 让 left跟进，即缩小窗口
        if(right-left+1-maxCount > k){ //需要替换的字符个数就是当前窗口的大小减去窗口中数量最多的字符的数量
            count[s[left]-'A']--; //减少对应字符的数量，有可能减的是数量最多的字符数量也有可能不是
            left++;
        }
        //当窗口内可替换的字符数小于等于k时，我们需要根据该窗口长度来确定是否更新result
        res=max(res, right - left + 1);
        right++;
    }

    return res;
}

// 8. 尽可能使字符串相等LC
// 题目描述：给你两个长度相同的字符串，s 和 t。将 s中的第i个字符变到t中的第 i 个字符需要|s[i] - t[i]|的开销（开销可能为 0）
//          用于变更字符串的最大预算是 maxCost。在转化字符串时，总开销应当小于等于该预算
//      如 输入：s = "abcd", t = "bcdf", maxCost = 3
//         输出：3， 解释：s 中的 "abc" 可以变为 "bcd"。开销为 3，所以最大长度为 3。
// 思路：滑动窗口：t 和 s 的相对应的子字符串做差值之后的 cost。
//      若窗口内的 cost 小于等于 maxCost 时，我们需要记录最长的子字符串；若 cost 大于 maxCost 的话，表示窗口溢出，我们需要缩小窗口了。
int equalSubstring(string s, string t, int maxCost) {
    int left=0,right=0;
    int res=0,cost=0,n=s.size();

    while(right<n){//固定窗口的边界
        cost+=abs(s[right]-t[right]);

        while(cost>maxCost){ //窗口溢出，需要缩小窗口
            cost-=abs(s[left]-t[left]);
            left++;
        }
        //窗内中cost小于等于maxCost时，我们需要记录最长的子字符串
        res=max(res, right - left + 1);
        right++;
    }
    return res;
}

// 9. 统计优美数组
// 题目描述：连续子数组中恰好有 k 个奇数数字，我们就认为这个子数组是「优美子数组」。
// 思路：滑动窗口
int numberOfSubarrays(vector<int>& nums, int k) {
    int left = 0, right = 0, oddCnt = 0, res = 0;
    while (right < nums.size()) {
        // 右指针先走，每遇到一个奇数则 oddCnt++。
        if ((nums[right++] & 1) == 1) {
            oddCnt++;
        }

        // 若当前滑动窗口中有 k 个奇数了
        if (oddCnt == k) {
            int tmp = right;
            // 将滑动窗口的右边界向右拓展，直到遇到下一个奇数（或出界）
            // 统计出右边界偶数的个数
            for(;right < nums.size() && (nums[right] & 1) == 0;right++);
            int rightEvenCnt = right - tmp; // rightEvenCnt 即为第 k 个奇数右边的偶数的个数

            // 接下来统计滑动窗口的左边界 偶数的个数
            // leftEvenCnt 即为第 1 个奇数左边的偶数的个数
            int leftEvenCnt = 0;
            while ((nums[left] & 1) == 0) {
                leftEvenCnt++;
                left++;
            }
            // 第 1 个奇数左边的 leftEvenCnt 个偶数都可以作为优美子数组的起点
            // (因为第1个奇数左边可以1个偶数都不取，所以起点的选择有 leftEvenCnt + 1 种）
            // 第 k 个奇数右边的 rightEvenCnt 个偶数都可以作为优美子数组的终点
            // (因为第k个奇数右边可以1个偶数都不取，所以终点的选择有 rightEvenCnt + 1 种）
            // 所以该滑动窗口中，优美子数组左右起点的选择组合数为 (leftEvenCnt + 1) * (rightEvenCnt + 1)
            res += (leftEvenCnt + 1) * (rightEvenCnt + 1);

            // 此时 left 指向的是第 1 个奇数，因为该区间已经统计完了，因此 left 右移一位，oddCnt--
            left++;
            oddCnt--;
        }

    }

    return res;
}//（时间复杂度 O(N)，空间复杂度 O(1)




// 10. 长度最小的和>=target的子数组
//  暴力法O(n^2)：初始化子数组的最小长度为无穷大，枚举数组nums 中的每个下标作为子数组的开始下标，
//      对于每个开始下标 i，需要找到大于或等于 i 的最小下标 j，使得从nums[i] 到 nums[j] 的元素和大于或等于 s，并更新子数组的最小长度（此时子数组的长度是 j-i+1j−i+1）
//  前缀和+二分O(n*log(n))：问题转换为在前缀和数组下标[0,i] 范围内找到满足「值小于等于s−t」的最大下标，充当子数组左端点的前一个值。
//      由于数据都是正数，presums单调递增，所以可以二分做
//  滑动窗口O(n)：先不断往右娜窗口直到当sum大于s，尝试往左收缩区间并更新最小长度
//      (PS: 这题如果不限制正数就变得非常难，lc862)
int minSubArrayLen_vpresum(int s, vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    int ans = INT_MAX;
    vector<int> sums(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        sums[i] = sums[i - 1] + nums[i - 1];
    }
    for (int i = 1; i <= n; i++) {
        int target = s + sums[i - 1];
        auto bound = lower_bound(sums.begin(), sums.end(), target);
        if (bound != sums.end()) {
            ans = min(ans, static_cast<int>((bound - sums.begin()) - (i - 1)));
        }
    }
    return ans == INT_MAX ? 0 : ans;
}
//滑动窗口
int minSubArrayLen(int s, vector<int>& nums) {
    int n = nums.size();
    if (n == 0) {
        return 0;
    }
    int ans = INT_MAX;

    int start = 0, right = 0;
    int sum = 0;
    while (right < n) {
        sum += nums[right];
        while (sum >= s) { // 收缩左区间
            ans = min(ans, right - start + 1); // 刷新满足sum>s的最小数组长度
            sum -= nums[start];
            start++;
        }
        right++;
    }
    return ans == INT_MAX ? 0 : ans;
} // 时间复杂度 O(n)

// 11. 和为target的连续正整数序列
// 输入：target = 9
// 输出：[[2,3,4],[4,5]]
//   滑动窗口
vector<vector<int>> findContinuousSequence(int target) {
    int l=1, r=2;
    int sum=1;
    vector<vector<int>> res;
    while(l <= target/2){ //  注意这里退出条件容易写错成： while(r <= (target+1)/2){
        sum += r;
        while(sum != target && l <= target/2){ // 这里也是，退出条件不是 r <= (target+1)/2
            if(sum < target){
                r++;
                sum += r;
            }else if(sum > target){
                sum -= l;
                l++;
            }
        }
        if(sum == target){
            vector<int> window(r-l+1);
            for(int i = l; i <=r; i++){
                window[i-l] = i;
            }
            res.push_back(window);
            sum-=l;
            l++;
            r++;
        }
    }
    return res;
} // 时间复杂度：O(target), 两个指针每个最多移动 target/2次
// 空间：O(1), 没有使用额外的空间
void test_findContinuousSequence(){
    vector<vector<int>> res;
    res = findContinuousSequence(9);
    for(const auto& nums:res){
        for(const auto& x:nums){
            cout<<x<<",";
        }
        cout<<endl;
    }
}

// 12. 乘积小于k的子数组
// 题目描述：给定一个正整数数组 nums和整数 k ，请找出该数组内乘积小于 k 的连续的子数组的个数。
// 思路：1<=nums[i]<=1000, 我们固定子数组 [i,j] 的右端点 j 时，显然左端点 i 越大，子数组元素乘积越小
// 我们可以从前往后处理所有的 nums[i]，枚举子数组的右端点 j，并且左端点从 i=0 开始；
// 每枚举一个右端点 jj，如果当前子数组元素乘积 prod 大于等于 k，那么我们右移左端点 i 直到满足当前子数组元素乘积小于 k 或者 i>j，那么元素乘积小于 k 的子数组数目为 j−i+1。返回
int numSubarrayProductLessThanK(vector<int>& nums, int k) {
    int n = nums.size(), ans = 0;
    if (k <= 1) return 0;
    int i= 0, prod = 1;
    for (int j = 0; j < n; j++) {
        prod *= nums[j];
        while (prod >= k) {
            prod /= nums[i ++];
        }
        ans += j - i  + 1;
    }
    return ans;
} // 时间复杂度：O(n)
// 空间复杂度：O(1)


// 13. 得到K个黑块的最少涂色次数
// 题目描述：blocks[i] 要么是 'W' 要么是 'B' ，表示第 i 块的颜色。字符 'W' 和 'B' 分别表示白色和黑色。
//      请返回至少出现 一次 连续 k 个黑色块的 最少 操作次数
// 思路：用一个固定大小为 kk 的「滑动窗口」表示出现连续 kk 个黑色块的区间，我们需要将该区间全部变为黑色块，
// 此时我们需要的操作次数为该区间中白色块的数目，那么我们只需要在「滑动窗口」从左向右移动的过程中维护窗口中白色块的数目
int minimumRecolors(string blocks, int k) {
    int l = 0, r = 0, cnt = 0;
    while (r < k) {
        cnt += blocks[r] == 'W' ? 1 : 0;
        r++;
    }
    int res = cnt;
    while (r < blocks.size()) {
        cnt += blocks[r] == 'W' ? 1 : 0;
        cnt -= blocks[l] == 'W' ? 1 : 0;
        res = min(res, cnt);
        l++;
        r++;
    }
    return res;
}



#endif //DATASTRUCT_ALGORITHM_SLIDING_WINDOW_H

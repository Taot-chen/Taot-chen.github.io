---
layout: post
title: python_read_txt
date: 2023-12-24
---

## Python TXT 文件读取

### 1、小文件读取

待读取文件比较小的时候，可以随便读取，问题不大

```python
file_path = "test.txt"

# 可以使用 readlines() 函数，一次将文件所有行全部读取进 list 中，再进行后续的操作
fp = open(file_path, "r")
lines = fp.readlines()
first_line = lines[0]
last_line = lines[-1]

# 也可以逐行读取文件
with open(file_path, "r") as fp:
    # first line
    line = fp.readline()
    cnt = 1
    while line:
        line = fp.readline()
        cnt += 1
    
```

### 2、大文件读取最后一行

当文件非常大的时候，需要读取文件的最后一行，或者是最后几行。如果使用前面小文件的两种随意的方法，会在空间和时间上有较大的成本。对于大文件，需要读取最后一行或者几行的时候，可以借助文件指针，通过从末尾开始倒着读取，并判断自己需要读取的行

```python
file_path = "test.txt"
with open(file_path, "r") as fp:
    # first line
    first_line = fp.readline()
    
    # 设置偏移量
    offset = -100
    while 1:
        # 文件指针移动到文件末尾(2)，并向前偏移 100 个字符
        fp.seek(offset, 2)
    	# 读取文件指针所在位置到文件末尾之间的内容
        lines = fp.readlines()
        # 通过判断是否至少有 2 行，这样可以确保最后一行是完整的
        # 同样地，如果需要读取文件的最后 n 行，只需要判断是是否有 n + 1 行，这样可以保证最后 n 行是完整的
        if len(lines) >= 2:
            last_line = lines[-1]
            # last_n_lines = lines[1, :]
            break
        # 如果无法保证最后 n 行是完整的，可以通过将偏移量翻倍，直到满足条件为止
        offset *= 2
    
```

### 3、读取任意文件的任意行

使用小文件的两种方法也可以读取文件的任意行，但是对于比较大的文件， 在时间和空间的上的开销可能会比较大。可以采用**二分法 + 大文件读取最后几行**结合的方法来实现，这样可以有较好的性能。当然，也可以使用 python 的 linecache 库的方法，可以很简单地使用，性能不太好

```python
file_path = "test.txt"
import linecache
# 读取第 1000000 行
line = linecache.getline(file_path, 1000000)
```


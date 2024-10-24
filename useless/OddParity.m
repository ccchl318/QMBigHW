% 定义方程
f = @(x) x^2 + x^2 * (cot(x))^2 - 115.6;

% 初始化解的存储数组
roots = [];

% 设置初始猜测值，x>0开始
initial_guesses = [3, 5.5]; % 从这些初始点开始寻找解

% 使用fzero函数求解
for i = 1:length(initial_guesses)
    x_root = fzero(f, initial_guesses(i));
    roots = [roots, x_root]; % 存储解
end

% 显示求解结果
disp('方程的前几个解为：');
disp(roots);

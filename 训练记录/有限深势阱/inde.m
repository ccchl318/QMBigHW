% 定义被积函数
f = @(x) sin(3.366 * x).*sin(3.366* x);

% 进行数值积分
result = integral(f, -1.7, 1.7);

% 显示结果
disp('积分结果为：');
disp(result);

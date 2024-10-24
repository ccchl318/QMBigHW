% 定义被积函数
f = @(x) exp(2*5.3571*x);

% 进行数值积分
result = integral(f, -Inf, -1.7 );

% 显示结果
disp('积分结果为：');
disp(result);

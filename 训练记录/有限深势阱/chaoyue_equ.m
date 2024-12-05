

% 定义x范围
x = 0:0.01:11;

% y = x * tan(x) 的函数
y1 = x .* tan(x);

% 处理不连续性：将 tan(x) 非常大的值设为 NaN
y1(abs(tan(x)) > 20) = NaN;  % 这里的 20 是一个阈值，可以根据需要调整

% 绘制 y = x * tan(x)
figure;
plot(x, y1);
hold on;

% 绘制圆 x^2 + y^2 = 115.6
theta = linspace(0, 2*pi, 1000); % 参数角度
r = sqrt(115.6); % 圆的半径
x_circle = r * cos(theta);
y_circle = r * sin(theta);

% 绘制圆
plot(x_circle, y_circle, 'r'); 

% 添加图例和标签，使用LaTeX格式
legend({'$\eta =\varepsilon  \tan(\varepsilon)$', '$\eta^2 + \varepsilon^2 = 115.6$'}, 'Interpreter', 'latex');
xlabel('$\varepsilon$','Interpreter','latex');
ylabel('$\eta$','Interpreter','latex');
grid on;
axis equal; % 保持x轴和y轴比例相同

% 限制坐标轴范围
xlim([0 11]);
ylim([0 11]);

hold off;

% 定义x范围
x = 0:0.01:11;

% y = x * -cot(x) 的函数
y1 = x .* -cot(x);

% 处理不连续性：将 cot(x) 非常大的值设为 NaN
y1(abs(cot(x)) > 20) = NaN;  % 设置一个阈值，例如20，避免画出不连续的竖线

% 绘制 y = x * -cot(x)
figure;
plot(x, y1);
hold on;

% 绘制圆 x^2 + y^2 = 115.6
theta = linspace(0, 2*pi, 1000); % 参数角度
r = sqrt(115.6); % 圆的半径
x_circle = r * cos(theta);
y_circle = r * sin(theta);

% 绘制圆
plot(x_circle, y_circle, 'r'); 

% 添加图例和标签，使用LaTeX格式
legend({'$\eta =-\varepsilon  \cot(\varepsilon)$', '$\eta^2 + \varepsilon^2 = 115.6$'}, 'Interpreter', 'latex');
xlabel('$\varepsilon$','Interpreter','latex');
ylabel('$\eta$','Interpreter','latex');
grid on;
axis equal; % 保持x轴和y轴比例相同

% 限制坐标轴范围
xlim([0 11]);
ylim([0 11]);

hold off;


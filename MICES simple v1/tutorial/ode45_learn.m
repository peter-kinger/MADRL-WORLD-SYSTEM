% 示例 1：简单匿名函数
square = @(x) x^2;
result = square(5); % result 将会是 25

% 示例 2：带有多个参数的匿名函数
add = @(a, b) a + b;
result = add(3, 4); % result 将会是 7

% 其实也就是其中的 a,b 是变量，随着输入不同会变化
% 而其他事参数所以不用单独使用句柄的相关用法


% 定义时间范围
tspan = [0 5];
% 定义初始条件
y0 = 1;

% 使用匿名函数句柄
[t, y] = ode45(@(t, y) simpleODE(t, y), tspan, y0);
% 返回的直接是每次的结果

% 绘制结果
plot(t, y);
xlabel('Time');
ylabel('y');
title('Solution of dy/dt = -2y');

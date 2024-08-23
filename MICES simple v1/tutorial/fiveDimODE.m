% 简化的代码：function dydt = fiveDimODE(t, y)
function dydt = fiveDimODE(y, a, b)
    dydt = zeros(5,1); % 初始化输出向量
    dydt(1) = a * y(1);
    dydt(2) = b * y(1) - y(2);
    dydt(3) = y(2) - y(3);
    dydt(4) = y(3) - y(4);
    dydt(5) = y(4) - y(5);
end

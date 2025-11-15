import Foundation

/// UnaryOp类表示一元操作符
public class UnaryOp {
    /// 操作符名称
    public let name: String
    
    /// 初始化方法
    /// - Parameter name: 操作符名称
    public init(name: String) {
        self.name = name
    }
    
    /// 调用方法，执行一元操作
    /// - Parameter operand: 操作数节点
    /// - Returns: 结果节点
    func call(_ operand: Node) -> Node {
        // 默认实现返回一个unary节点
        // 具体的子类可以重写此方法来实现特定的操作逻辑
        return .unary(
            name: name,
            shape: operand.shape(),
            dtype: getDataType(operand),
            operand: operand
        )
    }
    
    /// 获取节点的数据类型
    /// - Parameter node: 输入节点
    /// - Returns: 数据类型
    private func getDataType(_ node: Node) -> DataType {
        switch node {
        case .leaf(_, let dtype):
            return dtype
        case .unary(_, _, let dtype, _):
            return dtype
        case .binary(_, _, let dtype, _, _):
            return dtype
        }
    }
}
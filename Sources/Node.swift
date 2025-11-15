import Foundation

// Implement String.StringInterpolation for Shape
extension String.StringInterpolation {
    mutating func appendInterpolation(_ shape: Shape) {
        let description =
            "["
            + shape.map { dim in
                switch dim {
                case .Static(let value):
                    return String(value)
                case .Dynamic(let name):
                    return "<\(name)>"
                }
            }.joined(separator: ", ") + "]"
        appendLiteral(description)
    }
}

// Implement String.StringInterpolation for Node
extension String.StringInterpolation {
    mutating func appendInterpolation(_ node: Node) {
        switch node {
        case .leaf(_, let dtype):
            appendLiteral("Leaf[shape: \(node.shape()), dtype: \(dtype.rawValue)]")
        case .unary(let name, _, let dtype, _):
            appendLiteral("Unary[name: \(name), shape: \(node.shape()), dtype: \(dtype.rawValue)]")
        case .binary(let name, _, let dtype, _, _):
            appendLiteral("Binary[name: \(name), shape: \(node.shape()), dtype: \(dtype.rawValue)]")
        }
    }
}

// Create a type alias for [ShapeType]
public typealias Shape = [ShapeType]

public enum ShapeType {
    case Static(UInt64)
    case Dynamic(String)
}

public enum DataType: String, CaseIterable {
    case float32 = "Float32"
    case float16 = "Float16"
    case bfloat16 = "BFloat16"
    case int32 = "Int32"
    case int64 = "Int64"
    case uint32 = "UInt32"
    case uint64 = "UInt64"
    case bool = "Bool"
}

// Tensor enum with self-referencing capabilities and reference semantics
public indirect enum Node {
    case leaf(shape: Shape, dtype: DataType)
    case unary(name: String, shape: Shape, dtype: DataType, operand: Node)
    case binary(name: String, shape: Shape, dtype: DataType, left: Node, right: Node)

    static func scalar(dtype: DataType) -> Node {
        return .leaf(shape: [], dtype: dtype)
    }

    // Get rank (number of dimensions) of the tensor
    func ndim() -> Int {
        switch self {
        case .leaf(let shape, _):
            return shape.count
        case .unary(_, let shape, _, _):
            return shape.count
        case .binary(_, let shape, _, _, _):
            return shape.count
        }
    }

    func dtype() -> DataType {
        switch self {
        case .leaf(_, let dtype):
            return dtype
        case .unary(_, _, let dtype, _):
            return dtype
        case .binary(_, _, let dtype, _, _):
            return dtype
        }
    }

    func shape() -> Shape {
        switch self {
        case .leaf(let shape, _):
            return shape
        case .unary(_, let shape, _, _):
            return shape
        case .binary(_, let shape, _, _, _):
            return shape
        }

    }

}

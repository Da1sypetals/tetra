// Implement String.StringInterpolation for Shape
extension String.StringInterpolation {
    mutating func appendInterpolation(_ shape: Shape) {
        let description =
            "["
            + shape.map(\.description).joined(separator: ", ")
            + "]"
        appendLiteral(description)
    }
}

// Implement String.StringInterpolation for Node
extension String.StringInterpolation {
    mutating func appendInterpolation(_ node: Node) {
        appendLiteral(node.description)
    }
}

// Create a type alias for [ShapeType]
public typealias Shape = [ShapeType]

public enum ShapeType: CustomStringConvertible {
    case Static(UInt64)
    case Dynamic(String)

    public var description: String {
        switch self {
        case .Static(let value):
            return String(value)
        case .Dynamic(let name):
            return "<\(name)>"
        }
    }
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
public indirect enum Node: CustomStringConvertible {
    case leaf(shape: Shape, dtype: DataType)
    case unary(name: String, shape: Shape, dtype: DataType, operand: Node)
    case binary(name: String, shape: Shape, dtype: DataType, left: Node, right: Node)
    case ternary(name: String, shape: Shape, dtype: DataType, first: Node, second: Node, third: Node)

    static func scalar(dtype: DataType) -> Node {
        .leaf(shape: [], dtype: dtype)
    }

    // Get rank (number of dimensions) of the tensor
    var ndim: Int {
        switch self {
        case .leaf(let shape, _), .unary(_, let shape, _, _), .binary(_, let shape, _, _, _), .ternary(_, let shape, _, _, _, _):
            return shape.count
        }
    }

    var dtype: DataType {
        switch self {
        case .leaf(_, let dtype), .unary(_, _, let dtype, _), .binary(_, _, let dtype, _, _), .ternary(_, _, let dtype, _, _, _):
            return dtype
        }
    }

    var shape: Shape {
        switch self {
        case .leaf(let shape, _), .unary(_, let shape, _, _), .binary(_, let shape, _, _, _), .ternary(_, let shape, _, _, _, _):
            return shape
        }
    }

    public var description: String {
        switch self {
        case .leaf(_, let dtype):
            return "Leaf[shape: \(shape), dtype: \(dtype)]"
        case .unary(let name, _, let dtype, _):
            return "Unary[name: \(name), shape: \(shape), dtype: \(dtype)]"
        case .binary(let name, _, let dtype, _, _):
            return "Binary[name: \(name), shape: \(shape), dtype: \(dtype)]"
        case .ternary(let name, _, let dtype, _, _, _):
            return "Ternary[name: \(name), shape: \(shape), dtype: \(dtype)]"
        }
    }
}

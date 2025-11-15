import Foundation

/// TernaryOp class represents a ternary operation that can be applied to three tensors
public class TernaryOp {
    private let name: String
    private let firstDtype: DataType
    private let secondDtype: DataType
    private let thirdDtype: DataType
    private let outputDtype: DataType

    /// Initialize a TernaryOp with operation name and data types
    /// - Parameters:
    ///   - name: The name of the operation (e.g., "where", "select")
    ///   - firstDtype: The data type of the first input tensor
    ///   - secondDtype: The data type of the second input tensor
    ///   - thirdDtype: The data type of the third input tensor
    ///   - outputDtype: The data type of the output tensor
    public init(name: String, firstDtype: DataType, secondDtype: DataType, thirdDtype: DataType, outputDtype: DataType) {
        self.name = name
        self.firstDtype = firstDtype
        self.secondDtype = secondDtype
        self.thirdDtype = thirdDtype
        self.outputDtype = outputDtype
    }

    /// Apply the ternary operation to three input nodes
    /// - Parameters:
    ///   - first: The first input node
    ///   - second: The second input node
    ///   - third: The third input node
    /// - Returns: A new Node representing the result of the ternary operation
    /// - Throws: RuntimeError if the shapes are not broadcastable or data types don't match
    public func call(_ first: Node, _ second: Node, _ third: Node) -> Node {
        // Check if first input data type matches the expected first data type
        let actualFirstDtype = first.dtype
        if actualFirstDtype != self.firstDtype {
            fatalError(
                "First input data type mismatch for operation '\(self.name)': expected \(self.firstDtype.rawValue), got \(actualFirstDtype.rawValue)"
            )
        }

        // Check if second input data type matches the expected second data type
        let actualSecondDtype = second.dtype
        if actualSecondDtype != self.secondDtype {
            fatalError(
                "Second input data type mismatch for operation '\(self.name)': expected \(self.secondDtype.rawValue), got \(actualSecondDtype.rawValue)"
            )
        }

        // Check if third input data type matches the expected third data type
        let actualThirdDtype = third.dtype
        if actualThirdDtype != self.thirdDtype {
            fatalError(
                "Third input data type mismatch for operation '\(self.name)': expected \(self.thirdDtype.rawValue), got \(actualThirdDtype.rawValue)"
            )
        }

        // Get the shapes from the input nodes
        let firstShape = first.shape
        let secondShape = second.shape
        let thirdShape = third.shape

        // Try to broadcast the shapes
        guard let broadcastedShape = broadcast(firstShape, secondShape, thirdShape) else {
            // If shapes are not broadcastable, throw a runtime error
            fatalError(
                "Cannot broadcast shapes \(firstShape), \(secondShape), and \(thirdShape) for operation '\(name)'"
            )
        }

        // Create and return a new ternary node
        return Node.ternary(
            name: self.name,
            shape: broadcastedShape,
            dtype: self.outputDtype,
            first: first,
            second: second,
            third: third
        )
    }

    /// Get the name of the operation
    public func getName() -> String {
        return self.name
    }

    /// Get the first input data type
    public func getFirstDtype() -> DataType {
        return self.firstDtype
    }

    /// Get the second input data type
    public func getSecondDtype() -> DataType {
        return self.secondDtype
    }

    /// Get the third input data type
    public func getThirdDtype() -> DataType {
        return self.thirdDtype
    }

    /// Get the output data type
    public func getOutputDtype() -> DataType {
        return self.outputDtype
    }
}
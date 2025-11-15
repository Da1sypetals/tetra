import Foundation

/// UnaryOp class represents a unary operation that can be applied to a tensor
public class UnaryOp {
    private let name: String
    private let inputDtype: DataType
    private let outputDtype: DataType

    /// Initialize a UnaryOp with operation name and data types
    /// - Parameters:
    ///   - name: The name of the operation (e.g., "relu", "sigmoid")
    ///   - inputDtype: The data type of the input tensor
    ///   - outputDtype: The data type of the output tensor
    public init(name: String, inputDtype: DataType, outputDtype: DataType) {
        self.name = name
        self.inputDtype = inputDtype
        self.outputDtype = outputDtype
    }

    /// Apply the unary operation to an input node
    /// - Parameter input: The input node to apply the operation to
    /// - Returns: A new Node representing the result of the unary operation
    public func call(_ input: Node) -> Node {
        // Check if input data type matches the expected input data type
        let actualInputDtype = input.dtype()
        if actualInputDtype != self.inputDtype {
            fatalError(
                "Input data type mismatch for operation '\(self.name)': expected \(self.inputDtype.rawValue), got \(actualInputDtype.rawValue)"
            )
        }

        // Get the shape from the input node
        let inputShape = input.shape()

        // Create and return a new unary node
        return Node.unary(
            name: self.name,
            shape: inputShape,
            dtype: self.outputDtype,
            operand: input
        )
    }

    /// Get the name of the operation
    public func getName() -> String {
        return self.name
    }

    /// Get the input data type
    public func getInputDtype() -> DataType {
        return self.inputDtype
    }

    /// Get the output data type
    public func getOutputDtype() -> DataType {
        return self.outputDtype
    }
}

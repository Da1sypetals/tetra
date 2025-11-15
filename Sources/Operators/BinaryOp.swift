import Foundation

/// BinaryOp class represents a binary operation that can be applied to two tensors
public class BinaryOp {
    private let name: String
    private let leftDtype: DataType
    private let rightDtype: DataType
    private let outputDtype: DataType

    /// Initialize a BinaryOp with operation name and data types
    /// - Parameters:
    ///   - name: The name of the operation (e.g., "add", "mul")
    ///   - leftDtype: The data type of the left input tensor
    ///   - rightDtype: The data type of the right input tensor
    ///   - outputDtype: The data type of the output tensor
    public init(name: String, leftDtype: DataType, rightDtype: DataType, outputDtype: DataType) {
        self.name = name
        self.leftDtype = leftDtype
        self.rightDtype = rightDtype
        self.outputDtype = outputDtype
    }

    /// Apply the binary operation to two input nodes
    /// - Parameters:
    ///   - left: The left input node
    ///   - right: The right input node
    /// - Returns: A new Node representing the result of the binary operation
    /// - Throws: RuntimeError if the shapes are not broadcastable or data types don't match
    public func call(_ left: Node, _ right: Node) -> Node {
        // Check if left input data type matches the expected left data type
        guard left.dtype == self.leftDtype else {
            fatalError(
                "Left input data type mismatch for operation '\(self.name)': expected \(self.leftDtype), got \(left.dtype)"
            )
        }

        // Check if right input data type matches the expected right data type
        guard right.dtype == self.rightDtype else {
            fatalError(
                "Right input data type mismatch for operation '\(self.name)': expected \(self.rightDtype), got \(right.dtype)"
            )
        }

        // Try to broadcast the shapes
        guard let broadcastedShape = broadcast(left.shape, right.shape) else {
            // If shapes are not broadcastable, throw a runtime error
            fatalError(
                "Cannot broadcast shapes \(left.shape) and \(right.shape) for operation '\(name)'"
            )
        }

        // Create and return a new binary node
        return Node.binary(
            name: self.name,
            shape: broadcastedShape,
            dtype: self.outputDtype,
            left: left,
            right: right
        )
    }

    /// Get the name of the operation
    public func getName() -> String {
        return self.name
    }

    /// Get the left input data type
    public func getLeftDtype() -> DataType {
        return self.leftDtype
    }

    /// Get the right input data type
    public func getRightDtype() -> DataType {
        return self.rightDtype
    }

    /// Get the output data type
    public func getOutputDtype() -> DataType {
        return self.outputDtype
    }
}

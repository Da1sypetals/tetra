import Foundation

/// Broadcast function that takes arbitrary number of shapes and returns broadcasted shape if broadcastable, else nil
/// - Parameter shapes: Variable number of Shape arrays to broadcast
/// - Returns: Broadcasted shape if all shapes are compatible, otherwise nil
public func broadcast(_ shapes: Shape...) -> Shape? {
    // Handle edge case: no shapes provided
    guard !shapes.isEmpty else {
        return []
    }

    // Start with the first shape as the result
    var resultShape = shapes[0]

    // Iterate through the remaining shapes
    for i in 1..<shapes.count {
        let currentShape = shapes[i]

        // Try to broadcast resultShape with currentShape
        guard let broadcastedShape = broadcastTwoShapes(resultShape, currentShape) else {
            return nil  // Shapes are not compatible for broadcasting
        }

        resultShape = broadcastedShape
    }

    return resultShape
}

/// Helper function to broadcast two shapes
/// - Parameters:
///   - shape1: First shape
///   - shape2: Second shape
/// - Returns: Broadcasted shape if compatible, otherwise nil
private func broadcastTwoShapes(_ shape1: Shape, _ shape2: Shape) -> Shape? {
    let ndim1 = shape1.count
    let ndim2 = shape2.count
    let maxDim = max(ndim1, ndim2)

    var resultShape = [ShapeType]()

    // Iterate from right to left (least significant dimension to most)
    for i in 0..<maxDim {
        let idx1 = ndim1 - 1 - i
        let idx2 = ndim2 - 1 - i

        let dim1 = idx1 >= 0 ? shape1[idx1] : nil
        let dim2 = idx2 >= 0 ? shape2[idx2] : nil

        // Determine the broadcasted dimension
        let broadcastedDim: ShapeType?

        switch (dim1, dim2) {
        case (nil, let dim?):
            // Shape1 has fewer dimensions, use dimension from shape2
            broadcastedDim = dim
        case (let dim?, nil):
            // Shape2 has fewer dimensions, use dimension from shape1
            broadcastedDim = dim
        case (let d1?, let d2?):
            // Both shapes have this dimension, get compatible dimension
            broadcastedDim = getCompatibleDimension(d1, d2)
            if broadcastedDim == nil {
                return nil  // Dimensions are not compatible
            }
        case (nil, nil):
            // This shouldn't happen, but handle gracefully
            broadcastedDim = nil
        }

        // Prepend to result (since we're iterating from right to left)
        if let dim = broadcastedDim {
            resultShape.insert(dim, at: 0)
        }
    }

    return resultShape
}

/// Check if two dimensions are compatible for broadcasting and return the larger dimension
/// - Parameters:
///   - dim1: First dimension
///   - dim2: Second dimension
/// - Returns: The larger dimension if compatible, otherwise nil
private func getCompatibleDimension(_ dim1: ShapeType, _ dim2: ShapeType) -> ShapeType? {
    switch (dim1, dim2) {
    case (.Static(let v1), .Static(let v2)):
        // Static dimensions are compatible if they are equal or one is 1
        if v1 == v2 || v1 == 1 || v2 == 1 {
            // Return the larger static value
            return v1 >= v2 ? dim1 : dim2
        }
        return nil
    case (.Static(let v), .Dynamic):
        // Static dimension is compatible with dynamic if static is 1
        if v == 1 {
            return dim2
        }
        return nil
    case (.Dynamic, .Static(let v)):
        // Dynamic dimension is compatible with static if static is 1
        if v == 1 {
            return dim1
        }
        return nil
    case (.Dynamic(let name1), .Dynamic(let name2)):
        // Dynamic dimensions are compatible only if they have the same name
        if name1 == name2 {
            // For dynamic dimensions with the same name, return either one
            return dim1
        }
        return nil
    }
}

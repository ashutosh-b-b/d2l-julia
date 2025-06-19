struct LinearRegressionConcise{N} <: AbstractModel 
    net::N
end

function d2lai.forward(lr::LinearRegressionConcise, x)
    lr.net(x)
end

function d2lai.loss(lr::LinearRegressionConcise, y_pred, y)
    Flux.Losses.mse(y_pred, y)
end

def train_model(model, train_loader, optimizer, mse_loss, cross_entropy_loss, num_epochs=10):
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, numeric_values, unit_labels in train_loader:
            optimizer.zero_grad()  # Zero out gradients
            
            # Forward pass
            predicted_values, predicted_units = model(images)
            
            # Compute losses
            loss_value = mse_loss(predicted_values, numeric_values)
            loss_unit = cross_entropy_loss(predicted_units, unit_labels)
            total_loss = loss_value + loss_unit  # Total loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
        
# Train the model
train_model(model, train_loader, optimizer, mse_loss, cross_entropy_loss, num_epochs=10)

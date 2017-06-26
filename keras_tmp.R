input <- layer_input(shape=(maxlen))
embedding_weight = data.frame(word = names(tok$word_index), idx = t(data.frame(tok$word_index)))
embedding_weight = embedding_weight[order(embedding_weight$idx, decreasing = FALSE), ]
embedding_weight = merge(embedding_weight, word_vectors, by = 0, all.x = TRUE)
embedding_weight[is.na(embedding_weight)] <- 0
embedding_weight[, 1:3] <- NULL


model <- keras_model_sequential()
model %>%
  layer_embedding(length(tok$word_index) + 1, word_embedding_size, input_length = maxlen) %>%
  # layer_embedding(name = 'emb', length(tok$word_index) + 1, word_embedding_size,
  #                 weights=embedding_weight, input_length=maxlen, trainable=FALSE) %>%
  layer_dropout(0.2) %>%
  layer_conv_1d(
    filters = 250, kernel_size = 3,
    padding = "valid", activation = "relu", strides = 1
  ) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(encoding_dim) %>%
  layer_dropout(0.2) %>%
  layer_activation(name = "activation", "relu") %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")


model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

model %>%
  fit(
    x_train, label[1:nrow(x_train)],
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_test, label[(nrow(x_train)+1):nrow(data_idx)])
  )

K <- keras::backend()
get_activations <- K$`function`(list(get_layer(model, 'e1')$input, K$learning_phase()),
                                list(get_layer(model, 'e3')$output))
activations <- data.frame(get_activations(list(x_train, 1)))

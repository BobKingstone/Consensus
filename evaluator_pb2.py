# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: evaluator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x65valuator.proto\"4\n\x11\x45valuationRequest\x12\x0e\n\x06sample\x18\x01 \x01(\t\x12\x0f\n\x07ndarray\x18\x02 \x01(\x0c\"\x8e\x01\n\x12\x45valuationDecision\x12.\n\x08\x64\x65\x63ision\x18\x01 \x01(\x0e\x32\x1c.EvaluationDecision.Decision\x12\x0f\n\x07ndarray\x18\x02 \x01(\x0c\x12\x12\n\nexperience\x18\x03 \x01(\x05\"#\n\x08\x44\x65\x63ision\x12\n\n\x06\x42\x45NIGN\x10\x00\x12\x0b\n\x07\x41NOMALY\x10\x01\x32G\n\tEvaluator\x12:\n\rGetEvaluation\x12\x12.EvaluationRequest\x1a\x13.EvaluationDecision\"\x00\x62\x06proto3')



_EVALUATIONREQUEST = DESCRIPTOR.message_types_by_name['EvaluationRequest']
_EVALUATIONDECISION = DESCRIPTOR.message_types_by_name['EvaluationDecision']
_EVALUATIONDECISION_DECISION = _EVALUATIONDECISION.enum_types_by_name['Decision']
EvaluationRequest = _reflection.GeneratedProtocolMessageType('EvaluationRequest', (_message.Message,), {
  'DESCRIPTOR' : _EVALUATIONREQUEST,
  '__module__' : 'evaluator_pb2'
  # @@protoc_insertion_point(class_scope:EvaluationRequest)
  })
_sym_db.RegisterMessage(EvaluationRequest)

EvaluationDecision = _reflection.GeneratedProtocolMessageType('EvaluationDecision', (_message.Message,), {
  'DESCRIPTOR' : _EVALUATIONDECISION,
  '__module__' : 'evaluator_pb2'
  # @@protoc_insertion_point(class_scope:EvaluationDecision)
  })
_sym_db.RegisterMessage(EvaluationDecision)

_EVALUATOR = DESCRIPTOR.services_by_name['Evaluator']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _EVALUATIONREQUEST._serialized_start=19
  _EVALUATIONREQUEST._serialized_end=71
  _EVALUATIONDECISION._serialized_start=74
  _EVALUATIONDECISION._serialized_end=216
  _EVALUATIONDECISION_DECISION._serialized_start=181
  _EVALUATIONDECISION_DECISION._serialized_end=216
  _EVALUATOR._serialized_start=218
  _EVALUATOR._serialized_end=289
# @@protoc_insertion_point(module_scope)

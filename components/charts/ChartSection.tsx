"use client"

import { ChartItem } from "../charts/metrics/types"
import MetricBar from "./metrics/MetricBar"

export default function ChartSection({
  data,
}: {
  data: ChartItem[]
}) {
  return (
    <div className="space-y-10">
      <MetricBar dataKey="confidence_percent" data={data} />
      <MetricBar dataKey="latency_ms" data={data} />
      <MetricBar dataKey="entropy" data={data} />
      <MetricBar dataKey="stability" data={data} />
      <MetricBar dataKey="ram_delta_mb" data={data} />
    </div>
  )
}

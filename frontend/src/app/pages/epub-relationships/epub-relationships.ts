import {
  Component, OnInit, AfterViewInit, signal,
  ElementRef, ViewChild, NgZone
} from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import * as d3 from 'd3';

interface Relationship {
  id: number;
  character_1_name: string;
  character_2_name: string;
  relationship_type: string;
}

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  degree: number;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  type: string;
  char1: string;
  char2: string;
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  char1: string;
  char2: string;
  type: string;
}

const TYPE_COLORS: Record<string, string> = {
  family:        '#a0784a',
  ally:          '#3a7a58',
  enemy:         '#b94a3a',
  mentor:        '#4a6a9a',
  master_servant:'#7a4a9a',
  other:         '#8a8479',
};

@Component({
  selector: 'app-epub-relationships',
  imports: [CommonModule, RouterLink],
  templateUrl: './epub-relationships.html',
  styleUrl: './epub-relationships.css'
})
export class EpubRelationships implements OnInit, AfterViewInit {
  @ViewChild('svgEl') svgEl!: ElementRef<SVGElement>;

  relationships   = signal<Relationship[]>([]);
  relationshipTypes = signal<string[]>([]);
  loading         = signal(true);
  error           = signal<string | null>(null);
  tooltip         = signal<TooltipState>({ visible: false, x: 0, y: 0, char1: '', char2: '', type: '' });

  private data: Relationship[] = [];
  private rendered = false;

  constructor(
    private http: HttpClient,
    private route: ActivatedRoute,
    private zone: NgZone
  ) {}

  ngOnInit() {
    const id = this.route.snapshot.paramMap.get('id');
    this.http.get<Relationship[]>(`/api/epubs/${id}/relationships/`).subscribe({
      next: data => {
        this.data = data;
        this.relationships.set(data);
        const types = [...new Set(data.map(r => r.relationship_type))];
        this.relationshipTypes.set(types);
        this.loading.set(false);
        // render after view has updated
        setTimeout(() => this.renderGraph(), 50);
      },
      error: () => {
        this.error.set('Could not load relationships.');
        this.loading.set(false);
      },
    });
  }

  ngAfterViewInit() {
    if (!this.loading() && !this.rendered) this.renderGraph();
  }

  typeColor(type: string): string {
    return TYPE_COLORS[type] ?? '#8a8479';
  }

  formatType(type: string): string {
    return (type ?? '').replace(/_/g, ' ');
  }

  private renderGraph() {
    if (this.rendered || !this.svgEl) return;
    this.rendered = true;

    const el = this.svgEl.nativeElement;
    const W = el.parentElement!.clientWidth || 700;
    const H = Math.max(560, window.innerHeight * 0.65);

    // Build nodes & links
    const nodeMap = new Map<string, GraphNode>();
    const links: GraphLink[] = [];

    for (const r of this.data) {
      if (!nodeMap.has(r.character_1_name)) nodeMap.set(r.character_1_name, { id: r.character_1_name, label: r.character_1_name, degree: 0 });
      if (!nodeMap.has(r.character_2_name)) nodeMap.set(r.character_2_name, { id: r.character_2_name, label: r.character_2_name, degree: 0 });
      nodeMap.get(r.character_1_name)!.degree++;
      nodeMap.get(r.character_2_name)!.degree++;
      links.push({ source: r.character_1_name, target: r.character_2_name, type: r.relationship_type, char1: r.character_1_name, char2: r.character_2_name });
    }

    const nodes = Array.from(nodeMap.values());
    const maxDeg = Math.max(...nodes.map(n => n.degree));

    const nodeRadius = (d: GraphNode) => 5 + (d.degree / maxDeg) * 14;

    // SVG setup
    const svg = d3.select(el)
      .attr('width', W)
      .attr('height', H)
      .attr('viewBox', `0 0 ${W} ${H}`);

    // Zoom
    const g = svg.append('g');
    svg.call(
      d3.zoom<SVGElement, unknown>()
        .scaleExtent([0.3, 4])
        .on('zoom', (event) => g.attr('transform', event.transform))
    );

    // Arrow markers per type
    const types = [...new Set(this.data.map(r => r.relationship_type))];
    svg.append('defs').selectAll('marker')
      .data(types)
      .join('marker')
        .attr('id', t => `arrow-${t}`)
        .attr('viewBox', '0 -4 8 8')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('markerWidth', 5)
        .attr('markerHeight', 5)
        .attr('orient', 'auto')
      .append('path')
        .attr('d', 'M0,-4L8,0L0,4')
        .attr('fill', t => this.typeColor(t));

    // Simulation
    const simulation = d3.forceSimulation<GraphNode>(nodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(links).id(d => d.id).distance(100).strength(0.4))
      .force('charge', d3.forceManyBody().strength(-220))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collision', d3.forceCollide<GraphNode>().radius(d => nodeRadius(d) + 18));

    // Links
    const link = g.append('g').attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
        .attr('stroke', d => this.typeColor(d.type))
        .attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.55)
        .attr('marker-end', d => `url(#arrow-${d.type})`)
        .style('cursor', 'pointer')
        .on('mouseenter', (event, d) => {
          d3.select(event.currentTarget).attr('stroke-opacity', 1).attr('stroke-width', 2.5);
          const rect = el.getBoundingClientRect();
          const wrap = el.parentElement!.getBoundingClientRect();
          this.zone.run(() => this.tooltip.set({
            visible: true,
            x: event.clientX - wrap.left + 12,
            y: event.clientY - wrap.top - 12,
            char1: d.char1,
            char2: d.char2,
            type: d.type,
          }));
        })
        .on('mousemove', (event) => {
          const wrap = el.parentElement!.getBoundingClientRect();
          this.zone.run(() => this.tooltip.update(t => ({ ...t, x: event.clientX - wrap.left + 12, y: event.clientY - wrap.top - 12 })));
        })
        .on('mouseleave', (event) => {
          d3.select(event.currentTarget).attr('stroke-opacity', 0.55).attr('stroke-width', 1.5);
          this.zone.run(() => this.tooltip.update(t => ({ ...t, visible: false })));
        });

    // Node groups
    const node = g.append('g').attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
        .style('cursor', 'grab')
        .call(
          d3.drag<SVGGElement, GraphNode>()
            .on('start', (event, d) => {
              if (!event.active) simulation.alphaTarget(0.3).restart();
              d.fx = d.x; d.fy = d.y;
            })
            .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
            .on('end', (event, d) => {
              if (!event.active) simulation.alphaTarget(0);
              d.fx = null; d.fy = null;
            }) as any
        );

    // Node circles
    node.append('circle')
      .attr('r', d => nodeRadius(d))
      .attr('fill', '#f7f4ef')
      .attr('stroke', '#1a1814')
      .attr('stroke-width', 1.5);

    // Node labels
    node.append('text')
      .text(d => d.label)
      .attr('x', d => nodeRadius(d) + 5)
      .attr('y', '0.35em')
      .attr('font-family', "'Playfair Display', serif")
      .attr('font-size', d => `${Math.max(9, Math.min(13, 9 + d.degree))}px`)
      .attr('fill', '#1a1814')
      .attr('pointer-events', 'none');

    // Highlight connected edges on node hover
    node
      .on('mouseenter', (event, d) => {
        link
          .attr('stroke-opacity', l =>
            (l.source as GraphNode).id === d.id || (l.target as GraphNode).id === d.id ? 1 : 0.08
          )
          .attr('stroke-width', l =>
            (l.source as GraphNode).id === d.id || (l.target as GraphNode).id === d.id ? 2.5 : 1.5
          );
        node.select('circle')
          .attr('stroke-opacity', n => n.id === d.id ? 1 : 0.3)
          .attr('fill', n => n.id === d.id ? '#1a1814' : '#f7f4ef');
        node.select('text')
          .attr('fill-opacity', n =>
            n.id === d.id ||
            links.some(l => ((l.source as GraphNode).id === d.id && (l.target as GraphNode).id === n.id) ||
                             ((l.target as GraphNode).id === d.id && (l.source as GraphNode).id === n.id))
              ? 1 : 0.2
          );
      })
      .on('mouseleave', () => {
        link.attr('stroke-opacity', 0.55).attr('stroke-width', 1.5);
        node.select('circle').attr('stroke-opacity', 1).attr('fill', '#f7f4ef');
        node.select('text').attr('fill-opacity', 1);
      });

    // Tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as GraphNode).x!)
        .attr('y1', d => (d.source as GraphNode).y!)
        .attr('x2', d => (d.target as GraphNode).x!)
        .attr('y2', d => (d.target as GraphNode).y!);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
  }
}

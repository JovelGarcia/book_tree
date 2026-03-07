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

// Hover color for links — type-colored on hover only
const TYPE_COLORS: Record<string, string> = {
  family:         '#a0784a',
  ally:           '#3a7a58',
  enemy:          '#b94a3a',
  mentor:         '#4a6a9a',
  master_servant: '#7a4a9a',
  other:          '#888480',
};

// Default link color — near-white light gray, no hue
const LINK_DEFAULT = '#d8d5d0';
const LINK_DIM     = '#eeece9';

@Component({
  selector: 'app-epub-relationships',
  imports: [CommonModule, RouterLink],
  templateUrl: './epub-relationships.html',
  styleUrl: './epub-relationships.css'
})
export class EpubRelationships implements OnInit, AfterViewInit {
  @ViewChild('svgEl') svgEl!: ElementRef<SVGElement>;

  relationships     = signal<Relationship[]>([]);
  relationshipTypes = signal<string[]>([]);
  loading           = signal(true);
  error             = signal<string | null>(null);
  tooltip           = signal<TooltipState>({ visible: false, x: 0, y: 0, char1: '', char2: '', type: '' });

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
    return TYPE_COLORS[type] ?? TYPE_COLORS['other'];
  }

  formatType(type: string): string {
    return (type ?? '').replace(/_/g, ' ');
  }

  private renderGraph() {
    if (this.rendered || !this.svgEl) return;
    this.rendered = true;

    const el = this.svgEl.nativeElement;
    const W = el.parentElement!.clientWidth || 700;
    const H = Math.max(580, window.innerHeight * 0.68);

    // ── Build graph data ────────────────────────────────────────────────────
    const nodeMap = new Map<string, GraphNode>();
    const links: GraphLink[] = [];

    for (const r of this.data) {
      if (!nodeMap.has(r.character_1_name))
        nodeMap.set(r.character_1_name, { id: r.character_1_name, label: r.character_1_name, degree: 0 });
      if (!nodeMap.has(r.character_2_name))
        nodeMap.set(r.character_2_name, { id: r.character_2_name, label: r.character_2_name, degree: 0 });
      nodeMap.get(r.character_1_name)!.degree++;
      nodeMap.get(r.character_2_name)!.degree++;
      links.push({
        source: r.character_1_name,
        target: r.character_2_name,
        type: r.relationship_type,
        char1: r.character_1_name,
        char2: r.character_2_name,
      });
    }

    const nodes = Array.from(nodeMap.values());
    const maxDeg = Math.max(...nodes.map(n => n.degree), 1);

    const nodeRadius = (d: GraphNode) => 4 + (d.degree / maxDeg) * 13;

    // ── SVG setup ────────────────────────────────────────────────────────────
    const svg = d3.select(el)
      .attr('width', W)
      .attr('height', H)
      .attr('viewBox', `0 0 ${W} ${H}`);

    // Zoom
    const g = svg.append('g');
    svg.call(
      d3.zoom<SVGElement, unknown>()
        .scaleExtent([0.25, 5])
        .on('zoom', (event) => g.attr('transform', event.transform))
    );

    // No arrow markers — edges are undirected

    // ── Simulation — Obsidian graph view physics ─────────────────────────
    // Obsidian's graph breathes: clusters form distinct organic shapes,
    // nodes don't collapse together. Key tuning vs d3 defaults:
    //   • charge: strong enough repulsion (-160) to create breathing room
    //   • link: moderate distance (80px) + soft strength (0.4) so clusters
    //     spread into their own shapes rather than snapping tight
    //   • center: very weak (0.03) — just anchors the graph, doesn't pull in
    //   • collision: generous padding keeps labels readable
    //   • slower cooling (alphaDecay 0.015) gives more time to find natural layout
    const simulation = d3.forceSimulation<GraphNode>(nodes)
      .force('link',
        d3.forceLink<GraphNode, GraphLink>(links)
          .id(d => d.id)
          .distance(80)      // breathing room between connected nodes
          .strength(0.4)     // soft pull — clusters spread, not collapse
      )
      .force('charge',
        d3.forceManyBody<GraphNode>()
          .strength(-160)    // strong enough to push clusters apart organically
          .distanceMax(500)
          .distanceMin(10)
      )
      .force('center',
        d3.forceCenter(W / 2, H / 2).strength(0.25)  // enough to hold isolated nodes near the cluster
      )
      .force('collision',
        d3.forceCollide<GraphNode>()
          .radius(d => nodeRadius(d) + 12)
          .strength(0.8)
      )
      .alphaDecay(0.015)     // slow cooling — more time to find organic shape
      .velocityDecay(.85);

    // ── Links — flat light gray, colored only on hover ────────────────────
    const link = g.append('g').attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
        .attr('stroke', LINK_DEFAULT)
        .attr('stroke-width', 1.2)
        .attr('stroke-opacity', 1)
        .style('cursor', 'pointer')
        .on('mouseenter', (event, d) => {
          d3.select(event.currentTarget)
            .attr('stroke', TYPE_COLORS[d.type] ?? TYPE_COLORS['other'])
            .attr('stroke-width', 2);
          const wrap = el.parentElement!.getBoundingClientRect();
          this.zone.run(() => this.tooltip.set({
            visible: true,
            x: event.clientX - wrap.left + 14,
            y: event.clientY - wrap.top - 14,
            char1: d.char1, char2: d.char2, type: d.type,
          }));
        })
        .on('mousemove', (event) => {
          const wrap = el.parentElement!.getBoundingClientRect();
          this.zone.run(() =>
            this.tooltip.update(t => ({ ...t, x: event.clientX - wrap.left + 14, y: event.clientY - wrap.top - 14 }))
          );
        })
        .on('mouseleave', (event, d) => {
          d3.select(event.currentTarget)
            .attr('stroke', LINK_DEFAULT)
            .attr('stroke-width', 1.2);
          this.zone.run(() => this.tooltip.update(t => ({ ...t, visible: false })));
        });

    // ── Nodes ─────────────────────────────────────────────────────────────
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

    // Outer glow ring — neutral gray, faint
    node.append('circle')
      .attr('r', d => nodeRadius(d) + 4)
      .attr('fill', 'none')
      .attr('stroke', '#b0aba4')
      .attr('stroke-width', d => Math.max(1, d.degree / maxDeg * 3))
      .attr('stroke-opacity', 0.2)
      .attr('class', 'node-glow');

    // Main node circle — white fill, black stroke
    node.append('circle')
      .attr('r', d => nodeRadius(d))
      .attr('fill', '#ffffff')
      .attr('stroke', '#1a1814')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.85);

    // Label — black, centered below the node
    node.append('text')
      .text(d => d.label)
      .attr('x', 0)
      .attr('y', d => nodeRadius(d) + 13)
      .attr('text-anchor', 'middle')
      .attr('font-family', "'Playfair Display', serif")
      .attr('font-size', d => `${Math.max(9.5, Math.min(13, 9 + d.degree * 0.7))}px`)
      .attr('fill', '#1a1814')
      .attr('fill-opacity', 0.85)
      .attr('pointer-events', 'none');

    // ── Hover: dim everything not connected ───────────────────────────────
    node
      .on('mouseenter', (event, d) => {
        const connectedIds = new Set<string>([d.id]);
        links.forEach(l => {
          if ((l.source as GraphNode).id === d.id) connectedIds.add((l.target as GraphNode).id);
          if ((l.target as GraphNode).id === d.id) connectedIds.add((l.source as GraphNode).id);
        });

        link
          .attr('stroke', l =>
            (l.source as GraphNode).id === d.id || (l.target as GraphNode).id === d.id
              ? TYPE_COLORS[l.type] ?? TYPE_COLORS['other']
              : LINK_DIM
          )
          .attr('stroke-width', l =>
            (l.source as GraphNode).id === d.id || (l.target as GraphNode).id === d.id ? 2 : 1
          );

        node.select('circle + circle')  // main circle (second circle)
          .attr('stroke-opacity', (n: GraphNode) => connectedIds.has(n.id) ? 0.85 : 0.2);

        node.select('text')
          .attr('fill', (n: GraphNode) => n.id === d.id ? '#000000' : connectedIds.has(n.id) ? '#1a1814' : '#c8c4be')
          .attr('fill-opacity', 1);
      })
      .on('mouseleave', () => {
        link
          .attr('stroke', LINK_DEFAULT)
          .attr('stroke-width', 1.2);
        node.select('circle + circle')
          .attr('stroke-opacity', 0.85);
        node.select('text')
          .attr('fill', '#1a1814')
          .attr('fill-opacity', 0.85);
      });

    // ── Tick ─────────────────────────────────────────────────────────────
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
